package yencsimd

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"math/rand/v2"
	"strings"
	"testing"
	"testing/iotest"
)

// yencLines returns data yEnc encoded and dot-stuffed, ending in CRLF.
func yencLines(data []byte) string {
	art := encode(data, false)
	start := bytes.Index(art, []byte("\r\n")) + 2
	end := bytes.LastIndex(art, []byte("=yend"))
	return string(art[start:end])
}

func randomData(seed uint64, n int) []byte {
	r := rand.New(rand.NewPCG(seed, seed))
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(r.Uint32())
	}
	return b
}

func singlePart(data []byte, name string) string {
	return fmt.Sprintf("=ybegin line=128 size=%d name=%s\r\n%s=yend size=%d crc32=%08x\r\n.\r\n",
		len(data), name, yencLines(data), len(data), crc32.ChecksumIEEE(data))
}

func multiPart(file []byte, part, total int, begin, end int64) string {
	data := file[begin-1 : end]
	return fmt.Sprintf("=ybegin part=%d total=%d line=128 size=%d name=big.bin\r\n=ypart begin=%d end=%d\r\n%s=yend size=%d part=%d pcrc32=%08x crc32=%08x\r\n.\r\n",
		part, total, len(file), begin, end, yencLines(data), len(data), part, crc32.ChecksumIEEE(data), crc32.ChecksumIEEE(file))
}

// readers runs f with the article delivered whole and one byte at a time.
func readers(t *testing.T, article string, f func(t *testing.T, r io.Reader)) {
	t.Run("whole", func(t *testing.T) { f(t, strings.NewReader(article)) })
	t.Run("bytewise", func(t *testing.T) { f(t, iotest.OneByteReader(strings.NewReader(article))) })
}

func TestReaderSinglePart(t *testing.T) {
	data := randomData(1, 300_000)
	readers(t, singlePart(data, "my file.bin"), func(t *testing.T, r io.Reader) {
		d := NewReader(r)
		got, err := io.ReadAll(d)
		if err != nil || !bytes.Equal(got, data) {
			t.Fatalf("err=%v, %d bytes, equal=%v", err, len(got), bytes.Equal(got, data))
		}
		m := d.Meta()
		if m.Name != "my file.bin" || m.Size != int64(len(data)) || m.Part != 0 || m.Begin != 1 || m.End != int64(len(data)) || !m.HasCRC || m.CRC32 != crc32.ChecksumIEEE(data) {
			t.Fatalf("meta %+v", m)
		}
	})
}

func TestReaderMultiPart(t *testing.T) {
	file := randomData(2, 1000_000)
	readers(t, multiPart(file, 2, 3, 400_001, 800_000), func(t *testing.T, r io.Reader) {
		got, m, err := decodeFrom(r)
		if err != nil || !bytes.Equal(got, file[400_000:800_000]) {
			t.Fatalf("err=%v, %d bytes", err, len(got))
		}
		if m.Part != 2 || m.Total != 3 || m.Begin != 400_001 || m.End != 800_000 || m.Size != 1000_000 || m.CRC32 != crc32.ChecksumIEEE(got) {
			t.Fatalf("meta %+v", m)
		}
	})
}

func decodeFrom(r io.Reader) ([]byte, Meta, error) {
	d := NewReader(r)
	b, err := io.ReadAll(d)
	return b, d.Meta(), err
}

func TestReaderErrors(t *testing.T) {
	data := randomData(3, 50_000)
	good := singlePart(data, "x.bin")
	badCRC := strings.Replace(good, fmt.Sprintf("crc32=%08x", crc32.ChecksumIEEE(data)), "crc32=00000000", 1)
	badSize := strings.Replace(good, fmt.Sprintf("=yend size=%d", len(data)), "=yend size=1", 1)
	truncated := good[:strings.Index(good, "=yend")] + ".\r\n"
	for _, c := range []struct {
		name, article string
		want          error
	}{
		{"crc mismatch", badCRC, ErrCRC},
		{"size mismatch", badSize, ErrSize},
		{"no =yend", truncated, ErrTruncated},
		{"input ends inside the data", good[:len(good)/2], ErrTruncated},
		{"no =ybegin", "just text\r\n.\r\n", ErrNoYbegin},
		{"empty input", "", ErrNoYbegin},
	} {
		t.Run(c.name, func(t *testing.T) {
			readers(t, c.article, func(t *testing.T, r io.Reader) {
				if _, _, err := decodeFrom(r); !errors.Is(err, c.want) {
					t.Fatalf("err = %v, want %v", err, c.want)
				}
			})
		})
	}
}

// An ARTICLE response: headers and a blank line before the yEnc data.
func TestReaderSkipsHeaders(t *testing.T) {
	data := randomData(4, 10_000)
	article := "Subject: [1/1] \"x.bin\" yEnc (1/1)\r\nMessage-ID: <x@example.test>\r\n\r\n" + singlePart(data, "x.bin")
	got, _, err := Decode([]byte(article))
	if err != nil || !bytes.Equal(got, data) {
		t.Fatalf("err=%v", err)
	}
}

// After the article the connection is at the next response: the closing "."
// line is read, nothing after it.
func TestReaderStopsAfterTheArticle(t *testing.T) {
	data := randomData(5, 20_000)
	br := bufio.NewReader(strings.NewReader(singlePart(data, "x.bin") + "223 next response\r\n"))
	if _, err := io.ReadAll(NewReader(br)); err != nil {
		t.Fatal(err)
	}
	rest, _ := io.ReadAll(br)
	if string(rest) != "223 next response\r\n" {
		t.Fatalf("left on the connection: %q", rest)
	}
}

func TestReaderOnAllKernels(t *testing.T) {
	data := randomData(6, 200_000)
	article := singlePart(data, "k.bin")
	for _, k := range kernels() {
		withKernel(k, func() {
			if got, _, err := Decode([]byte(article)); err != nil || !bytes.Equal(got, data) {
				t.Fatalf("kernel %s: err=%v", k, err)
			}
		})
	}
}

// Decode must not panic or hang on any input.
func FuzzDecode(f *testing.F) {
	f.Add([]byte(singlePart([]byte("hello"), "a.bin")))
	f.Add([]byte(multiPart([]byte("hello world"), 1, 2, 1, 5)))
	f.Add([]byte("=ybegin part=1 size=5 name=x\r\n=ypart begin=1 end=5\r\n.\r\n"))
	f.Fuzz(func(t *testing.T, article []byte) {
		Decode(article)
	})
}

// Read decodes straight into p when p is big enough and copies out of an
// internal buffer when it is not: the output must be the same for any mix of
// sizes, with any bufio.Reader size, and the data of a truncated article must
// still come out before ErrTruncated.
func TestReaderReadSizes(t *testing.T) {
	data := randomData(7, 200_000)
	good := singlePart(data, "sizes.bin")
	truncated := good[:strings.Index(good, "=yend")] + ".\r\n"
	patterns := [][]int{{1}, {100}, {4096}, {5000}, {32 << 10}, {1 << 20}, {0, 1, 5000, 3, 70_000, 512, 4096}}
	buf := make([]byte, 1<<20)
	for _, bufSize := range []int{16, 4 << 10, 64 << 10} {
		for _, sizes := range patterns {
			for _, c := range []struct {
				article string
				want    error
			}{{good, io.EOF}, {truncated, ErrTruncated}} {
				d := NewReader(bufio.NewReaderSize(strings.NewReader(c.article), bufSize))
				var got []byte
				var err error
				for i := 0; err == nil; i++ {
					var n int
					n, err = d.Read(buf[:sizes[i%len(sizes)]])
					got = append(got, buf[:n]...)
				}
				if err != c.want || !bytes.Equal(got, data) {
					t.Fatalf("bufio %d, reads %v: err=%v, %d bytes, equal=%v", bufSize, sizes, err, len(got), bytes.Equal(got, data))
				}
				if m := d.Meta(); c.want == io.EOF && (!m.HasCRC || m.CRC32 != crc32.ChecksumIEEE(data)) {
					t.Fatalf("bufio %d, reads %v: meta %+v", bufSize, sizes, m)
				}
			}
		}
	}
}

// Decode returns the data it decoded along with an error about it.
func TestDecodeErrorsReturnData(t *testing.T) {
	data := randomData(8, 30_000)
	good := singlePart(data, "x.bin")
	badCRC := strings.Replace(good, fmt.Sprintf("crc32=%08x", crc32.ChecksumIEEE(data)), "crc32=00000000", 1)
	truncated := good[:strings.Index(good, "=yend")] + ".\r\n"
	for _, c := range []struct {
		name, article string
		want          error
	}{{"crc", badCRC, ErrCRC}, {"truncated", truncated, ErrTruncated}} {
		got, m, err := Decode([]byte(c.article))
		if !errors.Is(err, c.want) || !bytes.Equal(got, data) || m.Name != "x.bin" {
			t.Fatalf("%s: err=%v, %d bytes, equal=%v, meta %+v", c.name, err, len(got), bytes.Equal(got, data), m)
		}
	}
}
