package yencsimd

import (
	"bufio"
	"bytes"
	"errors"
	"io"
	"runtime"
	"strings"
	"testing"
	"testing/iotest"
)

// TestStreamEnd checks how the yEnc lines end: a =ybegin, =ypart or =yend
// line cut off by the end of the input (no line end) gives ErrTruncated,
// while an article that ends after a whole =yend line still decodes; only
// one line end is removed, so ".\r\r\n" is not the "." line; a =y line longer
// than maxControlLine gives ErrHeader. Each case goes through Decode and
// Reader (whole and byte by byte); when the article has its "." line, Reader
// must leave the connection at the next response with AtEnd true, also read
// a byte at a time through a 16-byte bufio.Reader.
func TestStreamEnd(t *testing.T) {
	data := randomData(12, 5000)
	good := singlePart(data, "x.bin")
	noDot := strings.TrimSuffix(good, ".\r\n")
	file := randomData(13, 3000)
	mp := multiPart(file, 2, 3, 1001, 2000)
	part := file[1000:2000]
	iy := strings.Index(good, "=yend")
	yend := good[iy : iy+strings.Index(good[iy:], "\r\n")]
	head := "=ybegin line=128 size=5000 name="
	sub := func(s, old, new string) string {
		if !strings.Contains(s, old) {
			t.Fatalf("test setup: %q not in article", old)
		}
		return strings.Replace(s, old, new, 1)
	}
	badBegin := strings.TrimSuffix(sub(good, "size=5000 ", "size=x "), ".\r\n")
	pad := " x=" + strings.Repeat("y", 70<<10)

	for _, c := range []struct {
		name    string
		article string
		want    error  // nil: decodes to data
		data    []byte // what a case that decodes gives
		dotEnd  bool   // the article ends with its "." line
		name2   string // when set: the Meta.Name a case that decodes gives
	}{
		// a =y line cut off by the end of the input
		{name: "input ends inside =ybegin, before size=", article: "=ybegin line=128 siz", want: ErrTruncated},
		{name: "input ends inside =ybegin, in name=", article: head + "x.b", want: ErrTruncated},
		{name: "input ends inside =ypart", article: mp[:strings.Index(mp, "=ypart")+len("=ypart begin=1001 en")], want: ErrTruncated},
		{name: "input ends inside =yend, before crc32=", article: good[:iy+len("=yend size=5000 ")], want: ErrTruncated},
		{name: "input ends inside =yend, in crc32=", article: good[:iy+len(yend)-3], want: ErrTruncated},
		{name: "input ends after a whole =yend line", article: noDot, data: data},
		{name: "input ends after a whole =yend line of a part", article: strings.TrimSuffix(mp, ".\r\n"), data: part},

		// one line end only
		{name: `".\r\r\n" before =ybegin is not the end`, article: ".\r\r\n" + good, data: data, dotEnd: true},
		{name: `".\r\r\n" after =yend is not the end`, article: noDot + ".\r\r\n.\r\n", data: data, dotEnd: true},
		{name: `".\r\r\n" after a bad =ybegin is not the end`, article: badBegin + ".\r\r\n.\r\n", want: ErrHeader, dotEnd: true},
		{name: "=y lines ending in CR CR LF", article: sub(sub(good, "name=x.bin\r\n", "name=x.bin\r\r\n"), yend+"\r\n", yend+"\r\r\n"), data: data, dotEnd: true, name2: "x.bin"},
		{name: "=y lines of a part ending in CR CR LF", article: strings.TrimSuffix(sub(sub(mp, "name=big.bin\r\n", "name=big.bin\r\r\n"), "end=2000\r\n", "end=2000\r\r\n"), "\r\n.\r\n") + "\r\r\n.\r\n", data: part, dotEnd: true, name2: "big.bin"},

		// long lines
		{name: "=ybegin of maxControlLine bytes", article: singlePart(data, strings.Repeat("n", maxControlLine-len(head))), data: data, dotEnd: true},
		{name: "=ybegin of maxControlLine+1 bytes", article: singlePart(data, strings.Repeat("n", maxControlLine+1-len(head))), want: ErrHeader, dotEnd: true},
		{name: "long =ypart", article: sub(mp, "=ypart begin=1001 end=2000", "=ypart begin=1001 end=2000"+pad), want: ErrHeader, dotEnd: true},
		{name: "long =yend", article: sub(good, yend, yend+pad), want: ErrHeader, dotEnd: true},
		{name: "long line before =ybegin", article: strings.Repeat("h", 200<<10) + "\r\n" + good, data: data, dotEnd: true},
		{name: "long line before =ybegin, then the end", article: strings.Repeat("h", 200<<10) + "\r\n.\r\n", want: ErrNoYbegin, dotEnd: true},
	} {
		t.Run(c.name, func(t *testing.T) {
			check := func(how string, got []byte, m Meta, err error) {
				t.Helper()
				switch {
				case c.want == nil && c.name2 != "" && m.Name != c.name2:
					t.Fatalf("%s: Name = %q, want %q", how, m.Name, c.name2)
				case c.want == nil && err != nil && err != io.EOF:
					t.Fatalf("%s: err = %v, want none", how, err)
				case c.want != nil && !errors.Is(err, c.want):
					t.Fatalf("%s: err = %v, want %v", how, err, c.want)
				case c.want == nil && !bytes.Equal(got, c.data):
					t.Fatalf("%s: %d bytes, want %d", how, len(got), len(c.data))
				}
			}
			got, m, err := Decode([]byte(c.article))
			check("Decode", got, m, err)
			readers(t, c.article, func(t *testing.T, r io.Reader) {
				d := NewReader(r)
				got, err := io.ReadAll(d)
				check("Reader", got, d.Meta(), err)
				if !d.AtEnd() {
					t.Fatalf("Reader: AtEnd() = false after err %v", err)
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
				got, err := io.ReadAll(d)
				check(conn.how, got, d.Meta(), err)
				if !d.AtEnd() {
					t.Fatalf("%s: AtEnd() = false after err %v", conn.how, err)
				}
				if rest, _ := io.ReadAll(conn.br); string(rest) != "223 next response\r\n" {
					t.Fatalf("%s: left on the connection: %.40q", conn.how, rest)
				}
			}
		})
	}
}

// TestLongLineMemory checks that a line far longer than Reader's buffer
// costs no more than about maxControlLine, wherever it comes: before
// =ybegin (dropped), as a =ybegin or =yend line (ErrHeader), or after =yend.
func TestLongLineMemory(t *testing.T) {
	data := randomData(14, 1000)
	good := singlePart(data, "x.bin")
	long := strings.Repeat("x", 4<<20)
	iy := strings.Index(good, "=yend")
	yend := good[iy : iy+strings.Index(good[iy:], "\r\n")]
	for _, c := range []struct {
		name, article string
		want          error // nil: decodes
	}{
		{"before =ybegin", long + "\r\n" + good, nil},
		{"as the =ybegin line", strings.Replace(good, "name=x.bin", "name="+long, 1), ErrHeader},
		{"as the =yend line", strings.Replace(good, yend, yend+" x="+long, 1), ErrHeader},
		{"after =yend", strings.TrimSuffix(good, ".\r\n") + long + "\r\n.\r\n", nil},
	} {
		t.Run(c.name, func(t *testing.T) {
			br := bufio.NewReaderSize(strings.NewReader(c.article+"223 next response\r\n"), 4<<10)
			d := NewReader(br)
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			got, err := io.ReadAll(d)
			runtime.ReadMemStats(&after)
			if c.want == nil && (err != nil || !bytes.Equal(got, data)) || c.want != nil && !errors.Is(err, c.want) {
				t.Fatalf("err = %v (%d bytes), want %v", err, len(got), c.want)
			}
			if rest, _ := io.ReadAll(br); string(rest) != "223 next response\r\n" {
				t.Fatalf("left on the connection: %.40q", rest)
			}
			if n := after.TotalAlloc - before.TotalAlloc; n > 1<<20 {
				t.Fatalf("Reader allocated %d bytes", n)
			}
		})
	}
}
