package yencsimd

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"strconv"
)

// Errors returned by Reader and Decode.
var (
	// ErrNoYbegin: the article ended before a "=ybegin" line.
	ErrNoYbegin = errors.New("yencsimd: no =ybegin line")
	// ErrTruncated: the article ended after "=ybegin" but before "=yend"
	// (also where a multi-part post needs its "=ypart" line).
	ErrTruncated = errors.New("yencsimd: article ends before =yend")
	// ErrSize: the decoded size differs from the =ybegin/=ypart/=yend sizes.
	ErrSize = errors.New("yencsimd: size mismatch")
	// ErrCRC: the CRC32 of the decoded data differs from the =yend line.
	ErrCRC = errors.New("yencsimd: crc32 mismatch")
	// ErrHeader: =ybegin has no size=, =ypart has no begin= or end=, a
	// =ybegin, =ypart or =yend line has an invalid value (size, part, begin,
	// end, or the crc32/pcrc32 that is checked), or another line comes where
	// a multi-part post needs =ypart. part=, crc32=, pcrc32= and =yend size=
	// may be left out; an invalid total= is ignored.
	ErrHeader = errors.New("yencsimd: bad =y line")
)

// Meta is what the yEnc header and trailer lines say about the data.
// Name, Size, Part, Total, Begin and End are set once Read returned the first
// data (or an error); CRC32 and HasCRC once Read returned io.EOF.
type Meta struct {
	Name  string // file name (=ybegin name=)
	Size  int64  // size of the whole file (=ybegin size=)
	Part  int    // part number (=ybegin part=), 0 for a single-part post
	Total int    // number of parts (=ybegin total=), 0 if not given
	Begin int64  // first byte of this part in the file, 1-based (=ypart begin=); 1 for single-part
	End   int64  // last byte of this part, inclusive (=ypart end=); Size for single-part

	CRC32  uint32 // CRC32 the trailer gives for this part (pcrc32=, or crc32= for single-part)
	HasCRC bool   // whether the trailer had one; without it the CRC isn't checked
}

// Reader decodes one yEnc article from an NNTP connection or a buffer.
//
// It expects the article the way it comes off the wire after a "222" (BODY)
// or "220" (ARTICLE) status line: lines end in CRLF, lines starting with "."
// are dot-stuffed, and the article ends with a line holding a single ".".
// Header lines and anything else before "=ybegin" are skipped. Do not put a
// net/textproto DotReader in front of it: that already removes the
// dot-stuffing, and Reader would remove it a second time.
//
// Read returns the decoded data. At the end it checks the sizes and the
// CRC32 and returns io.EOF, or ErrSize / ErrCRC / ErrTruncated / ErrNoYbegin /
// ErrHeader, or a plain error for another =y line inside the data. By then it
// has read through the closing "." line (or to the end of the input), so an
// NNTP connection is ready for its next command. Only a read error from the
// underlying reader leaves the connection inside the article; Read returns it,
// also when it comes while reading on to the "." line (then joined with the
// error above, if any, so errors.Is finds both).
type Reader struct {
	r    *bufio.Reader
	meta Meta
	err  error // sticky: returned once out is drained

	started   bool // the =ybegin (and =ypart) lines are read
	inData    bool // in the yEnc data, between the header lines and =yend
	truncated bool // the article ended inside the data
	state     State
	crc       uint32 // CRC32 of the decoded bytes so far
	n         int64  // decoded bytes so far

	out []byte // decoded into dst, not yet returned
	dst []byte // output buffer for Read calls with a small p
}

// readerBufSize is the buffer NewReader puts on a plain io.Reader.
const readerBufSize = 64 << 10

// directMin is the len(p) from which Read decodes straight into p even if
// that splits what is buffered into more decoder calls: a call costs about
// as much as copying 4-8 KiB (measured with AVX2 and AVX-512), so smaller
// pieces are cheaper to copy out of one big decode.
const directMin = 8 << 10

// NewReader returns a Reader that decodes one article from r. If r is a
// *bufio.Reader it is used directly (net/textproto's Reader.R is one), so no
// bytes of the next response are read ahead and lost.
//
// Each step decodes what the bufio.Reader has buffered, so its buffer size
// sets how much work a step does and how often the connection is read: for
// an NNTP connection give textproto a bufio.NewReaderSize(conn, 64<<10)
// rather than the 4 KiB default; Reader never reads past the article.
func NewReader(r io.Reader) *Reader {
	br, ok := r.(*bufio.Reader)
	if !ok {
		br = bufio.NewReaderSize(r, readerBufSize)
	}
	return &Reader{r: br, state: StateCRLF}
}

// Meta returns what the yEnc lines said so far (see Meta).
func (d *Reader) Meta() Meta { return d.meta }

// Read implements io.Reader. When p can take all that is buffered, or holds
// 8 KiB or more, it decodes straight into p, so io.Copy (32 KiB pieces) and
// io.ReadAll (once its buffer has grown) get the data without a copy. Read may
// use p[n:] as scratch space, as io.Reader allows: bytes of p past n are not
// preserved.
func (d *Reader) Read(p []byte) (int, error) {
	for len(d.out) == 0 {
		if d.err != nil {
			return 0, d.err
		}
		var n int
		n, d.err = d.step(p)
		if n > 0 {
			return n, nil
		}
	}
	n := copy(p, d.out)
	d.out = d.out[n:]
	return n, nil
}

// step reads and decodes the next piece of input, into p (returning how many
// bytes) if p can take a worthwhile piece, else into d.out; or it returns
// the final error.
func (d *Reader) step(p []byte) (int, error) {
	if !d.started {
		return 0, d.readHeader()
	}
	if !d.inData {
		return 0, d.finish()
	}
	// decode what the bufio.Reader has buffered (at least one byte)
	if _, err := d.r.Peek(1); err != nil {
		return 0, d.eof(err)
	}
	src, _ := d.r.Peek(d.r.Buffered())
	// into p when it takes the whole piece (no copy, no extra call) or a
	// piece worth a call of its own
	direct := len(p) >= len(src) || len(p) >= directMin
	var dst []byte
	if direct {
		// the output of a piece is never longer than the piece
		src = src[:min(len(src), len(p))]
		dst = p[:len(src)]
	} else {
		if cap(d.dst) < len(src) {
			d.dst = make([]byte, len(src))
		}
		dst = d.dst[:len(src)]
	}
	nd, ns, end, err := DecodeIncremental(dst, src, &d.state)
	if err != nil {
		return 0, err
	}
	d.r.Discard(ns)
	d.crc = crc32.Update(d.crc, crc32.IEEETable, dst[:nd])
	d.n += int64(nd)
	switch end {
	case EndControl: // "=y" read: the rest of the line is "end ..." (or another =y line)
		d.inData = false
	case EndArticle: // ".\r\n" before =yend: return the data, then ErrTruncated
		d.inData, d.truncated = false, true
	}
	if direct {
		return nd, nil
	}
	d.out = dst[:nd]
	return 0, nil
}

// readHeader skips lines up to "=ybegin", parses it and an "=ypart" line.
func (d *Reader) readHeader() error {
	for {
		line, err := d.readLine()
		if err != nil {
			return d.eof(err)
		}
		if bytes.Equal(line, []byte(".")) {
			return ErrNoYbegin
		}
		if !bytes.HasPrefix(line, []byte("=ybegin ")) {
			continue
		}
		if err := d.meta.parseBegin(line); err != nil {
			return d.drain(err)
		}
		if d.meta.Part > 0 {
			part, err := d.readLine()
			if err == io.EOF {
				return ErrTruncated // =ybegin seen: the article was cut short
			}
			if err != nil {
				return err
			}
			if bytes.Equal(part, []byte(".")) {
				return ErrTruncated
			}
			if err := d.meta.parsePart(part); err != nil {
				return d.drain(err)
			}
		}
		d.started, d.inData = true, true
		return nil
	}
}

// finish parses the =yend line, checks size and CRC and reads through the
// closing "." line.
func (d *Reader) finish() error {
	if d.truncated {
		return ErrTruncated
	}
	rest, err := d.readLine() // after "=y": "end size=... pcrc32=..."
	if err != nil {
		return d.eof(err)
	}
	endSize, err := d.meta.parseEnd(rest) // also refuses another =y line in the data
	if err == nil {
		err = d.meta.check(d.n, d.crc, endSize)
	}
	if err := d.drain(err); err != nil { // a plain file may just end
		return err
	}
	return io.EOF
}

// eof turns the input ending early into the right error.
func (d *Reader) eof(err error) error {
	if err != io.EOF {
		return err
	}
	if !d.started {
		return ErrNoYbegin
	}
	return ErrTruncated
}

// drain reads through the closing "." line (or to the end of the input), so
// that after the =yend line, or an error in the yEnc lines, the connection is
// at the next response. It returns err, or a read error that stopped it
// (joined with err if both): then the connection is still inside the article.
func (d *Reader) drain(err error) error {
	for {
		line, rerr := d.readLine()
		switch {
		case rerr == io.EOF:
			return err
		case rerr != nil && err == nil:
			return rerr
		case rerr != nil:
			return errors.Join(err, rerr)
		case bytes.Equal(line, []byte(".")):
			return err
		}
	}
}

// readLine returns the next line without its CR LF (it may span buffers).
func (d *Reader) readLine() ([]byte, error) {
	line, err := d.r.ReadSlice('\n')
	if err == bufio.ErrBufferFull {
		long := append([]byte(nil), line...)
		for err == bufio.ErrBufferFull {
			line, err = d.r.ReadSlice('\n')
			long = append(long, line...)
		}
		line = long
	}
	if err != nil && (err != io.EOF || len(line) == 0) {
		return nil, err
	}
	return bytes.TrimRight(line, "\r\n"), nil
}

// cutLine is readLine on a slice: the first line of b without its CR LF and
// what follows it. ok is false at the end of b; like readLine, a last line
// without LF is still a line.
func cutLine(b []byte) (line, rest []byte, ok bool) {
	if len(b) == 0 {
		return nil, nil, false
	}
	i := bytes.IndexByte(b, '\n')
	if i < 0 {
		i = len(b) - 1
	}
	return bytes.TrimRight(b[:i+1], "\r\n"), b[i+1:], true
}

// parseBegin reads an =ybegin line. name= comes last and runs to the end of
// the line (it may hold spaces and "part=" etc.), so the other keys are looked
// up only before it.
func (m *Meta) parseBegin(line []byte) error {
	if i := bytes.Index(line, []byte(" name=")); i >= 0 {
		m.Name = string(line[i+len(" name="):])
		line = line[:i]
	}
	size, ok := intField(line, "size", 63)
	if !ok || size < 0 {
		return fmt.Errorf("%w: =ybegin without a valid size= in %q", ErrHeader, truncate(line))
	}
	m.Size = size
	// part= is optional; when given it counts from 1
	if _, has := field(line, "part"); has {
		part, ok := intField(line, "part", 31)
		if !ok || part < 1 {
			return fmt.Errorf("%w: =ybegin part= not a part number in %q", ErrHeader, truncate(line))
		}
		m.Part = int(part)
	}
	// total= is only passed on to the caller: one that is not a part count
	// counts as not given
	if total, ok := intField(line, "total", 31); ok && total >= 1 {
		m.Total = int(total)
	}
	m.Begin, m.End = 1, m.Size
	return nil
}

// parsePart sets m from the line after =ybegin of a part, which must be
// "=ypart ...".
func (m *Meta) parsePart(line []byte) error {
	if !bytes.HasPrefix(line, []byte("=ypart ")) {
		return fmt.Errorf("%w: multi-part post without =ypart line, got %q", ErrHeader, truncate(line))
	}
	// begin..end is inclusive and 1-based; end = begin-1 is an empty part
	// (a 0-byte file posted as part 1 of 1: begin=1 end=0)
	begin, okb := intField(line, "begin", 63)
	end, oke := intField(line, "end", 63)
	if !okb || !oke || begin < 1 || end < begin-1 || end > m.Size {
		return fmt.Errorf("%w: =ypart begin/end outside 1..%d in %q", ErrHeader, m.Size, truncate(line))
	}
	m.Begin, m.End = begin, end
	return nil
}

// isEnd reports whether rest, an =y line without its "=y", is an =yend line.
func isEnd(rest []byte) bool {
	return bytes.Equal(rest, []byte("end")) || bytes.HasPrefix(rest, []byte("end "))
}

// parseEnd reads an =yend line (without its "=y"). A size= or crc32=/pcrc32=
// that is given must be valid; endSize is -1 when size= is not given.
func (m *Meta) parseEnd(rest []byte) (endSize int64, err error) {
	if !isEnd(rest) {
		// another =y line inside the data (=ypart repeated?): not supported
		return 0, fmt.Errorf("yencsimd: unexpected line =y%s", truncate(rest))
	}
	key := "crc32"
	if m.Part > 0 {
		key = "pcrc32"
	}
	if v, ok := field(rest, key); ok {
		// hex digits, upper or lower case, maybe fewer than 8; a "0x" in
		// front is not yEnc but harmless, so it is accepted
		if len(v) > 2 && v[0] == '0' && (v[1] == 'x' || v[1] == 'X') {
			v = v[2:]
		}
		c, err := strconv.ParseUint(string(v), 16, 32)
		if err != nil {
			return 0, fmt.Errorf("%w: =yend %s=%q is not a CRC32", ErrHeader, key, truncate(v))
		}
		m.CRC32, m.HasCRC = uint32(c), true
	}
	endSize = -1 // not given
	if _, ok := field(rest, "size"); ok {
		n, ok := intField(rest, "size", 63)
		if !ok || n < 0 {
			return 0, fmt.Errorf("%w: =yend size= not a size in %q", ErrHeader, truncate(rest))
		}
		endSize = n
	}
	return endSize, nil
}

// check compares the decoded size n and CRC with the header and trailer.
func (m *Meta) check(n int64, crc uint32, endSize int64) error {
	want := m.Size
	if m.Part > 0 {
		want = m.End - m.Begin + 1
	}
	if n != want || endSize >= 0 && endSize != n {
		return fmt.Errorf("%w: decoded %d bytes, header says %d, trailer %d", ErrSize, n, want, endSize)
	}
	if m.HasCRC && crc != m.CRC32 {
		return fmt.Errorf("%w: got %08x, trailer says %08x", ErrCRC, crc, m.CRC32)
	}
	return nil
}

// field returns the value of " key=" in a yEnc line (up to the next space).
func field(line []byte, key string) ([]byte, bool) {
	for _, sep := range []string{" ", ""} {
		k := []byte(sep + key + "=")
		i := bytes.Index(line, k)
		if i < 0 || (sep == "" && i != 0) {
			continue
		}
		v := line[i+len(k):]
		if j := bytes.IndexByte(v, ' '); j >= 0 {
			v = v[:j]
		}
		return v, true
	}
	return nil, false
}

// intField returns the decimal value of key= in line, which must fit in a
// signed integer of bits+1 bits; ok is false when the key is missing or its
// value is not such a number.
func intField(line []byte, key string, bits int) (n int64, ok bool) {
	v, has := field(line, key)
	if !has {
		return 0, false
	}
	n, err := strconv.ParseInt(string(v), 10, bits+1)
	return n, err == nil
}

func truncate(b []byte) []byte {
	if len(b) > 60 {
		return b[:60]
	}
	return b
}

// Decode decodes one yEnc article held in memory (as Reader expects it: CRLF
// lines, dot-stuffed, optionally with headers before =ybegin) and checks its
// size and CRC32. It returns the same errors as Reader, with the data decoded
// up to the error.
//
// The data is decoded in one step into a buffer as long as the rest of the
// article after the header lines (the output is never longer), so the returned
// slice has some spare capacity: about 3% for a typical post.
func Decode(article []byte) ([]byte, Meta, error) {
	var m Meta
	rest := article
	for {
		line, more, ok := cutLine(rest)
		if !ok {
			return []byte{}, m, ErrNoYbegin
		}
		rest = more
		if bytes.Equal(line, []byte(".")) {
			return []byte{}, m, ErrNoYbegin
		}
		if !bytes.HasPrefix(line, []byte("=ybegin ")) {
			continue
		}
		if err := m.parseBegin(line); err != nil {
			return []byte{}, m, err
		}
		if m.Part > 0 {
			part, more, ok := cutLine(rest)
			if !ok || bytes.Equal(part, []byte(".")) {
				return []byte{}, m, ErrTruncated // =ybegin seen: the article was cut short
			}
			rest = more
			if err := m.parsePart(part); err != nil {
				return []byte{}, m, err
			}
		}
		break
	}
	data := make([]byte, len(rest))
	st := StateCRLF
	nd, ns, end, _ := DecodeIncremental(data, rest, &st) // len(dst) == len(src): no error
	data = data[:nd]
	if end != EndControl { // "\r\n.\r\n", or the article ends, before =yend
		return data, m, ErrTruncated
	}
	line, _, ok := cutLine(rest[ns:]) // after "=y": "end size=... pcrc32=..."
	if !ok {
		return data, m, ErrTruncated
	}
	endSize, err := m.parseEnd(line)
	if err != nil {
		return data, m, err
	}
	var crc uint32
	if m.HasCRC {
		crc = crc32.ChecksumIEEE(data)
	}
	return data, m, m.check(int64(nd), crc, endSize)
}
