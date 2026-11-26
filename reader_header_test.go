package yencsimd

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"runtime"
	"strings"
	"testing"
	"testing/iotest"
)

// errAny marks a case where any error except ErrHeader is right.
var errAny = errors.New("any error")

// TestHeaderValues checks the values of the =ybegin, =ypart and =yend lines:
// keys inside name= are not read, missing or invalid values give ErrHeader,
// odd but harmless forms still decode, and a multi-part post cut off right
// after =ybegin gives ErrTruncated. Each case goes through Decode and Reader
// (whole and byte by byte); when the article has its closing "." line,
// Reader must leave the connection at the next response, also when it is
// read a byte at a time through a 16-byte bufio.Reader.
func TestHeaderValues(t *testing.T) {
	data := randomData(7, 5000)
	good := singlePart(data, "x.bin")
	file := randomData(8, 3000)
	mp := multiPart(file, 2, 3, 1001, 2000)
	part := file[1000:2000]
	sum := crc32.ChecksumIEEE(data)
	crc := fmt.Sprintf("crc32=%08x", sum)
	pcrc := fmt.Sprintf("pcrc32=%08x", crc32.ChecksumIEEE(part))
	sub := func(s, old, new string) string {
		if !strings.Contains(s, old) {
			t.Fatalf("test setup: %q not in article", old)
		}
		return strings.Replace(s, old, new, 1)
	}
	beginLine := fmt.Sprintf("=ybegin line=128 size=%d ", len(data))
	ypart := "=ypart begin=1001 end=2000"
	mpBegin := "=ybegin part=2 total=3 line=128 size=3000 name=big.bin"

	for _, c := range []struct {
		name    string
		article string
		want    error  // nil: decodes to data; errAny: any error but ErrHeader
		data    []byte // what a case that decodes gives
		meta    func(m Meta) bool
		dotEnd  bool // the article ends with its "." line
	}{
		// odd but harmless: these decode
		{name: "part= and total= inside name=", article: singlePart(data, "archive part=2 of 3 total=5.rar"), data: data, dotEnd: true,
			meta: func(m Meta) bool { return m.Name == "archive part=2 of 3 total=5.rar" && m.Part == 0 && m.Total == 0 }},
		{name: "size= inside name=", article: singlePart(data, "a size=99 b.bin"), data: data, dotEnd: true,
			meta: func(m Meta) bool { return m.Name == "a size=99 b.bin" && m.Size == int64(len(data)) }},
		{name: "0-byte file", article: singlePart(nil, "empty.bin"), data: []byte{}, dotEnd: true,
			meta: func(m Meta) bool { return m.Size == 0 && m.Begin == 1 && m.End == 0 }},
		{name: "upper-case crc32=", article: sub(good, crc, fmt.Sprintf("crc32=%08X", sum)), data: data, dotEnd: true},
		{name: "crc32= without leading zeros", article: sub(good, crc, fmt.Sprintf("crc32=%x", sum)), data: data, dotEnd: true},
		{name: "crc32= with 0x", article: sub(good, crc, fmt.Sprintf("crc32=0x%08x", sum)), data: data, dotEnd: true,
			meta: func(m Meta) bool { return m.HasCRC && m.CRC32 == sum }},
		{name: "part with pcrc32= and crc32=", article: mp, data: part, dotEnd: true,
			meta: func(m Meta) bool { return m.Part == 2 && m.Total == 3 && m.Begin == 1001 && m.End == 2000 }},
		{name: "part= without total=", article: sub(mp, " total=3 ", " "), data: part, dotEnd: true,
			meta: func(m Meta) bool { return m.Part == 2 && m.Total == 0 }},
		{name: "total=0 counts as not given", article: sub(mp, " total=3 ", " total=0 "), data: part, dotEnd: true,
			meta: func(m Meta) bool { return m.Total == 0 }},
		{name: "0-byte file as part 1 of 1", article: "=ybegin part=1 total=1 line=128 size=0 name=e.bin\r\n=ypart begin=1 end=0\r\n=yend size=0 part=1 pcrc32=00000000\r\n.\r\n", data: []byte{}, dotEnd: true,
			meta: func(m Meta) bool { return m.Part == 1 && m.Begin == 1 && m.End == 0 && m.Size == 0 }},

		// missing or invalid values: ErrHeader
		{name: "no size=", article: sub(good, beginLine, "=ybegin line=128 "), want: ErrHeader, dotEnd: true},
		{name: "negative size=", article: sub(good, beginLine, "=ybegin line=128 size=-5 "), want: ErrHeader, dotEnd: true},
		{name: "size= not a number", article: sub(good, beginLine, "=ybegin line=128 size=12x "), want: ErrHeader, dotEnd: true},
		{name: "part=0", article: sub(mp, "=ybegin part=2 ", "=ybegin part=0 "), want: ErrHeader, dotEnd: true},
		{name: "part=-1", article: sub(mp, "=ybegin part=2 ", "=ybegin part=-1 "), want: ErrHeader, dotEnd: true},
		{name: "part= beyond 31 bits", article: sub(mp, "=ybegin part=2 ", "=ybegin part=4294967298 "), want: ErrHeader, dotEnd: true},
		{name: "no begin=", article: sub(mp, ypart, "=ypart end=1000"), want: ErrHeader, dotEnd: true},
		{name: "begin=0", article: sub(mp, ypart, "=ypart begin=0 end=999"), want: ErrHeader, dotEnd: true},
		{name: "begin= after end=", article: sub(mp, ypart, "=ypart begin=2000 end=1001"), want: ErrHeader, dotEnd: true},
		{name: "end= after size=", article: sub(mp, ypart, "=ypart begin=2002 end=3001"), want: ErrHeader, dotEnd: true},
		{name: "no =ypart line", article: sub(mp, ypart+"\r\n", ""), want: ErrHeader, dotEnd: true},
		{name: "begin= far beyond size=", article: sub(mp, ypart, "=ypart begin=1000000000000001 end=1000000000001000"), want: ErrHeader, dotEnd: true},
		{name: "crc32= not hex", article: sub(good, crc, "crc32=zzzzzzzz"), want: ErrHeader, dotEnd: true},
		{name: "crc32= empty", article: sub(good, crc, "crc32="), want: ErrHeader, dotEnd: true},
		{name: "crc32= beyond 32 bits", article: sub(good, crc, "crc32=123456789"), want: ErrHeader, dotEnd: true},
		{name: "pcrc32= not hex", article: sub(mp, pcrc, "pcrc32=xyz"), want: ErrHeader, dotEnd: true},
		{name: "=yend size= not a number", article: sub(good, fmt.Sprintf("=yend size=%d", len(data)), "=yend size=abc"), want: ErrHeader, dotEnd: true},

		// other =y lines, and posts cut off after =ybegin
		{name: "=yending is not =yend", article: sub(good, "=yend size=", "=yending size="), want: errAny, dotEnd: true},
		{name: "name= before part=: =ypart lands in the data", article: sub(mp, mpBegin, "=ybegin line=128 size=3000 name=big.bin part=2 total=3"), want: errAny, dotEnd: true},
		{name: "multi-part, input ends after =ybegin", article: mp[:strings.Index(mp, "=ypart")], want: ErrTruncated},
		{name: "multi-part, . line after =ybegin", article: mp[:strings.Index(mp, "=ypart")] + ".\r\n", want: ErrTruncated, dotEnd: true},
	} {
		t.Run(c.name, func(t *testing.T) {
			check := func(how string, err error) {
				t.Helper()
				switch {
				case c.want == nil && err != nil && err != io.EOF:
					t.Fatalf("%s: err = %v, want none", how, err)
				case c.want == errAny && (err == nil || err == io.EOF || errors.Is(err, ErrHeader)):
					t.Fatalf("%s: err = %v, want an error other than ErrHeader", how, err)
				case c.want != nil && c.want != errAny && !errors.Is(err, c.want):
					t.Fatalf("%s: err = %v, want %v", how, err, c.want)
				}
			}
			got, m, err := Decode([]byte(c.article))
			check("Decode", err)
			if c.want == nil && (!bytes.Equal(got, c.data) || c.meta != nil && !c.meta(m)) {
				t.Fatalf("Decode: %d bytes (want %d), meta %+v", len(got), len(c.data), m)
			}
			readers(t, c.article, func(t *testing.T, r io.Reader) {
				got, m, err := decodeFrom(r)
				check("Reader", err)
				if c.want == nil && (!bytes.Equal(got, c.data) || c.meta != nil && !c.meta(m)) {
					t.Fatalf("Reader: %d bytes (want %d), meta %+v", len(got), len(c.data), m)
				}
			})
			if !c.dotEnd {
				return
			}
			for _, conn := range []struct {
				how string
				br  *bufio.Reader
			}{
				{"Reader, connection", bufio.NewReader(strings.NewReader(c.article + "223 next response\r\n"))},
				{"Reader, connection, bytewise", bufio.NewReaderSize(iotest.OneByteReader(strings.NewReader(c.article+"223 next response\r\n")), 16)},
			} {
				d := NewReader(conn.br)
				_, err := io.ReadAll(d)
				check(conn.how, err)
				if !d.AtEnd() {
					t.Fatalf("%s: AtEnd() = false after err %v", conn.how, err)
				}
				if rest, _ := io.ReadAll(conn.br); string(rest) != "223 next response\r\n" {
					t.Fatalf("%s: left on the connection: %q", conn.how, rest)
				}
			}
		})
	}
}

// TestDrainReadError checks that a read error while Reader reads on to the
// closing "." line (after =yend, or after an error in the yEnc lines) is
// returned, joined with the yEnc error if there is one, and not hidden behind
// io.EOF or that error: the connection is then still inside the article.
func TestDrainReadError(t *testing.T) {
	errRead := errors.New("read timeout")
	data := randomData(9, 5000)
	good := singlePart(data, "x.bin")
	noDot := strings.TrimSuffix(good, ".\r\n")
	file := randomData(10, 3000)
	mp := multiPart(file, 2, 3, 1001, 2000)
	for _, c := range []struct {
		name, input string
		want        error  // besides errRead; nil: only errRead
		text        string // and a plain error with this text
	}{
		{name: "after a good =yend", input: noDot},
		{name: "after a bad crc32=", input: strings.Replace(noDot, fmt.Sprintf("crc32=%08x", crc32.ChecksumIEEE(data)), "crc32=00000000", 1), want: ErrCRC},
		{name: "after =yending", input: strings.Replace(noDot, "=yend size=", "=yending size=", 1), text: "unexpected line =yending"},
		{name: "in the data after a bad size=", input: strings.Replace(good, "=ybegin line=128 size=5000 ", "=ybegin line=128 size=x ", 1)[:len(good)/2], want: ErrHeader},
		{name: "in the data after a bad =ypart", input: strings.Replace(mp, "=ypart begin=1001 end=2000", "=ypart begin=0 end=999", 1)[:len(mp)/2], want: ErrHeader},
		// not in the drain, but AtEnd must be false there too
		{name: "before =ybegin", input: "Subject: x\r\n"},
		{name: "on the =ypart line", input: mp[:strings.Index(mp, "=ypart")+5]},
		{name: "in the data", input: good[:len(good)/2]},
		{name: "on the =yend line", input: good[:strings.Index(good, "=yend")+6]},
	} {
		t.Run(c.name, func(t *testing.T) {
			for _, r := range []struct {
				how string
				r   io.Reader
			}{
				{"whole", io.MultiReader(strings.NewReader(c.input), iotest.ErrReader(errRead))},
				{"bytewise", iotest.OneByteReader(io.MultiReader(strings.NewReader(c.input), iotest.ErrReader(errRead)))},
			} {
				d := NewReader(r.r)
				_, err := io.ReadAll(d)
				if d.AtEnd() {
					t.Fatalf("%s: AtEnd() = true after err %v", r.how, err)
				}
				if !errors.Is(err, errRead) || c.want != nil && !errors.Is(err, c.want) ||
					err != nil && !strings.Contains(err.Error(), c.text) {
					t.Fatalf("%s: err = %v, want %v and %v %q", r.how, err, errRead, c.want, c.text)
				}
			}
		})
	}
}

// TestDrainLongLine checks that Reader reads on to the "." line past a line
// far longer than its buffer without collecting it: after a bad =ybegin
// (where the data that follows is skipped) and after =yend (a long trailer
// line), a 4 MiB line must not cost 4 MiB.
func TestDrainLongLine(t *testing.T) {
	data := randomData(11, 1000)
	good := singlePart(data, "x.bin")
	long := strings.Repeat("x", 4<<20) + "\r\n"
	noDot := strings.TrimSuffix(good, ".\r\n")
	badBegin := strings.Replace(noDot, "=ybegin line=128 size=1000 ", "=ybegin line=128 size=x ", 1)
	if badBegin == noDot {
		t.Fatal("test setup: =ybegin line not found")
	}
	for _, c := range []struct {
		name, article string
		want          error // nil: decodes
	}{
		{"after a bad =ybegin", badBegin + long + ".\r\n", ErrHeader},
		{"after =yend", noDot + long + ".\r\n", nil},
	} {
		t.Run(c.name, func(t *testing.T) {
			br := bufio.NewReaderSize(strings.NewReader(c.article+"223 next response\r\n"), 4<<10)
			d := NewReader(br)
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			_, err := io.ReadAll(d)
			runtime.ReadMemStats(&after)
			if c.want == nil && err != nil || c.want != nil && !errors.Is(err, c.want) {
				t.Fatalf("err = %v, want %v", err, c.want)
			}
			if !d.AtEnd() {
				t.Fatal("AtEnd() = false")
			}
			if rest, _ := io.ReadAll(br); string(rest) != "223 next response\r\n" {
				t.Fatalf("left on the connection: %q", rest)
			}
			if n := after.TotalAlloc - before.TotalAlloc; n > 1<<20 {
				t.Fatalf("reading on to the . line allocated %d bytes", n)
			}
		})
	}
}
