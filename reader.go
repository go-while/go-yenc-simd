package yencsimd

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"hash"
	"hash/crc32"
	"io"
	"strconv"
)

// Errors returned by Reader and Decode.
var (
	// ErrNoYbegin: the article ended before a "=ybegin" line.
	ErrNoYbegin = errors.New("yencsimd: no =ybegin line")
	// ErrTruncated: the article ended inside the yEnc data, before "=yend".
	ErrTruncated = errors.New("yencsimd: article ends before =yend")
	// ErrSize: the decoded size differs from the =ybegin/=ypart/=yend sizes.
	ErrSize = errors.New("yencsimd: size mismatch")
	// ErrCRC: the CRC32 of the decoded data differs from the =yend line.
	ErrCRC = errors.New("yencsimd: crc32 mismatch")
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
// CRC32 and returns io.EOF, or ErrSize / ErrCRC / ErrTruncated / ErrNoYbegin.
// By then it has read through the closing "." line (or to the end of the
// input), so an NNTP connection is ready for its next command.
type Reader struct {
	r    *bufio.Reader
	meta Meta
	err  error // sticky: returned once out is drained

	started   bool // the =ybegin (and =ypart) lines are read
	inData    bool // in the yEnc data, between the header lines and =yend
	truncated bool // the article ended inside the data
	state     State
	crc       hash.Hash32
	n         int64 // decoded bytes so far

	out []byte // decoded, not yet returned
	dst []byte
}

// NewReader returns a Reader that decodes one article from r. If r is a
// *bufio.Reader it is used directly (net/textproto's Reader.R is one), so no
// bytes of the next response are read ahead and lost.
func NewReader(r io.Reader) *Reader {
	br, ok := r.(*bufio.Reader)
	if !ok {
		br = bufio.NewReaderSize(r, 64<<10)
	}
	return &Reader{r: br, state: StateCRLF, crc: crc32.NewIEEE()}
}

// Meta returns what the yEnc lines said so far (see Meta).
func (d *Reader) Meta() Meta { return d.meta }

// Read implements io.Reader.
func (d *Reader) Read(p []byte) (int, error) {
	for len(d.out) == 0 {
		if d.err != nil {
			return 0, d.err
		}
		d.err = d.step()
	}
	n := copy(p, d.out)
	d.out = d.out[n:]
	return n, nil
}

// step reads and decodes the next piece of input into d.out, or returns the
// final error.
func (d *Reader) step() error {
	if !d.started {
		return d.readHeader()
	}
	if !d.inData {
		return d.finish()
	}
	// decode what the bufio.Reader has buffered (at least one byte)
	if _, err := d.r.Peek(1); err != nil {
		return d.eof(err)
	}
	src, _ := d.r.Peek(d.r.Buffered())
	if cap(d.dst) < len(src) {
		d.dst = make([]byte, len(src))
	}
	dst := d.dst[:len(src)]
	nd, ns, end, err := DecodeIncremental(dst, src, &d.state)
	if err != nil {
		return err
	}
	d.r.Discard(ns)
	d.crc.Write(dst[:nd])
	d.n += int64(nd)
	d.out = dst[:nd]
	switch end {
	case EndControl: // "=y" read: the rest of the line is "end ..." (or another =y line)
		d.inData = false
	case EndArticle: // ".\r\n" before =yend: return the data, then ErrTruncated
		d.inData, d.truncated = false, true
	}
	return nil
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
		d.meta.Name = string(nameField(line))
		d.meta.Size = intField(line, "size")
		d.meta.Part = int(intField(line, "part"))
		d.meta.Total = int(intField(line, "total"))
		d.meta.Begin, d.meta.End = 1, d.meta.Size
		if d.meta.Part > 0 {
			part, err := d.readLine()
			if err != nil {
				return d.eof(err)
			}
			if !bytes.HasPrefix(part, []byte("=ypart ")) {
				return fmt.Errorf("yencsimd: multi-part post without =ypart line, got %q", truncate(part))
			}
			d.meta.Begin = intField(part, "begin")
			d.meta.End = intField(part, "end")
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
	if !bytes.HasPrefix(rest, []byte("end")) {
		// another =y line inside the data (=ypart repeated?): not supported
		return fmt.Errorf("yencsimd: unexpected line =y%s", truncate(rest))
	}
	key := "crc32"
	if d.meta.Part > 0 {
		key = "pcrc32"
	}
	if v, ok := field(rest, key); ok {
		if c, err := strconv.ParseUint(string(v), 16, 32); err == nil {
			d.meta.CRC32, d.meta.HasCRC = uint32(c), true
		}
	}
	endSize := int64(-1) // not given
	if _, ok := field(rest, "size"); ok {
		endSize = intField(rest, "size")
	}
	err = d.check(endSize)
	// read through the end of the article; a plain file may just end
	for {
		line, rerr := d.readLine()
		if rerr != nil || bytes.Equal(line, []byte(".")) {
			break
		}
	}
	if err != nil {
		return err
	}
	return io.EOF
}

func (d *Reader) check(endSize int64) error {
	want := d.meta.Size
	if d.meta.Part > 0 {
		want = d.meta.End - d.meta.Begin + 1
	}
	if d.n != want || endSize >= 0 && endSize != d.n {
		return fmt.Errorf("%w: decoded %d bytes, header says %d, trailer %d", ErrSize, d.n, want, endSize)
	}
	if d.meta.HasCRC && d.crc.Sum32() != d.meta.CRC32 {
		return fmt.Errorf("%w: got %08x, trailer says %08x", ErrCRC, d.crc.Sum32(), d.meta.CRC32)
	}
	return nil
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

func intField(line []byte, key string) int64 {
	v, ok := field(line, key)
	if !ok {
		return 0
	}
	n, _ := strconv.ParseInt(string(v), 10, 64)
	return n
}

// nameField returns the name: the last field, which may contain spaces.
func nameField(line []byte) []byte {
	if i := bytes.Index(line, []byte(" name=")); i >= 0 {
		return line[i+len(" name="):]
	}
	return nil
}

func truncate(b []byte) []byte {
	if len(b) > 60 {
		return b[:60]
	}
	return b
}

// Decode decodes one yEnc article held in memory (as Reader expects it: CRLF
// lines, dot-stuffed, optionally with headers before =ybegin) and checks its
// size and CRC32.
func Decode(article []byte) ([]byte, Meta, error) {
	d := NewReader(bytes.NewReader(article))
	data, err := io.ReadAll(d)
	return data, d.Meta(), err
}
