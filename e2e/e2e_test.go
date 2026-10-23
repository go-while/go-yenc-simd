package e2e

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"net"
	"net/textproto"
	"strings"
	"sync"
	"testing"
	"time"

	yencsimd "github.com/go-while/go-yenc-simd"
	"github.com/go-while/nntp-testkit/fakeserver"
	"github.com/go-while/nntp-testkit/yenc"
	"github.com/mnightingale/rapidyenc"
)

// sizes are the data sizes of a generated group's posts: post n has sizeOf(n) bytes. They
// straddle the vector widths (16, 32, 64), the line length, the decoders' small-input
// cut-offs and a 4 KiB buffer, and end with a typical 750 KB post.
var sizes = []int{0, 1, 2, 3, 15, 16, 17, 63, 64, 65, 127, 128, 129, 130, 255, 256, 257,
	1000, 4095, 4096, 4097, 65536, 65543, 750_000}

func sizeOf(n int64) int { return sizes[(n-1)%int64(len(sizes))] }

// groups are the generated groups: every data profile, both encoder styles for '.', tab and
// space at line edges, short and long lines, and an encoder that escapes more than it must.
var groups = []struct {
	name string
	p    yenc.Profile
	o    yenc.Options
}{
	{"yenc.random", yenc.Random, yenc.Options{}},
	{"yenc.plain", yenc.Plain, yenc.Options{}},
	{"yenc.escapes", yenc.Escapes, yenc.Options{}},
	{"yenc.edges", yenc.Edges, yenc.Options{}},
	{"yenc.edges.raw", yenc.Edges, yenc.Options{RawDots: true, RawSpaces: true}},
	{"yenc.line64", yenc.Random, yenc.Options{Line: 64}},
	{"yenc.line997", yenc.Random, yenc.Options{Line: 997}},
	{"yenc.extra", yenc.Edges, yenc.Options{ExtraEscapes: "\t. "}},
}

func newStore() *fakeserver.Store {
	st := fakeserver.NewStore()
	for _, g := range groups {
		st.AddGenerated(g.name, "yEnc test posts", yenc.Generator(g.p, 1, int64(len(sizes)), sizeOf, g.o))
	}
	return st
}

// client is one NNTP connection, used the way the README shows: textproto for the commands,
// and its bufio.Reader for Reader.
type client struct {
	nc net.Conn
	r  *textproto.Reader
	w  *textproto.Writer
}

func dial(addr string, bufSize int) (*client, error) {
	nc, err := net.DialTimeout("tcp", addr, 10*time.Second)
	if err != nil {
		return nil, err
	}
	_ = nc.SetDeadline(time.Now().Add(5 * time.Minute)) // hang guard
	c := &client{nc, textproto.NewReader(bufio.NewReaderSize(nc, bufSize)), textproto.NewWriter(bufio.NewWriter(nc))}
	if _, _, err := c.r.ReadCodeLine(200); err != nil {
		nc.Close()
		return nil, fmt.Errorf("greeting: %w", err)
	}
	return c, nil
}

func mustDial(t testing.TB, addr string, bufSize int) *client {
	t.Helper()
	c, err := dial(addr, bufSize)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.nc.Close() })
	return c
}

func (c *client) cmd(code int, format string, args ...any) error {
	if err := c.w.PrintfLine(format, args...); err != nil {
		return err
	}
	if _, msg, err := c.r.ReadCodeLine(code); err != nil {
		return fmt.Errorf("%s: %w (%s)", fmt.Sprintf(format, args...), err, msg)
	}
	return nil
}

func (c *client) group(t testing.TB, name string) {
	t.Helper()
	if err := c.cmd(211, "GROUP %s", name); err != nil {
		t.Fatal(err)
	}
}

// body fetches post n of the current group with BODY and decodes it with Reader.
func (c *client) body(n int64) ([]byte, yencsimd.Meta, error) {
	if err := c.cmd(222, "BODY %d", n); err != nil {
		return nil, yencsimd.Meta{}, err
	}
	d := yencsimd.NewReader(c.r.R)
	var buf bytes.Buffer
	_, err := io.Copy(&buf, d)
	return buf.Bytes(), d.Meta(), err
}

// check compares what body returned for post n with the data the generator made.
func check(data []byte, m yencsimd.Meta, err error, p yenc.Profile, n int64, size int) error {
	want := yenc.Data(p, size, uint64(n))
	switch {
	case err != nil:
		return fmt.Errorf("post %d (%d bytes): %w", n, size, err)
	case !bytes.Equal(data, want):
		return fmt.Errorf("post %d: %d bytes decoded, want %d; first difference at byte %d", n, len(data), len(want), firstDiff(data, want))
	case m.Name != fmt.Sprintf("%s-%d.bin", p, n) || m.Size != int64(size) || m.Part != 0 ||
		m.Begin != 1 || m.End != int64(size) || !m.HasCRC || m.CRC32 != crc32.ChecksumIEEE(want):
		return fmt.Errorf("post %d: meta %+v", n, m)
	}
	return nil
}

func firstDiff(a, b []byte) int {
	for i := range min(len(a), len(b)) {
		if a[i] != b[i] {
			return i
		}
	}
	return min(len(a), len(b))
}

// eachKernel runs f as a subtest for each decode kernel of this CPU (Kernels) and restores the
// default afterwards. SetKernel is global, so these tests do not run in parallel.
func eachKernel(t *testing.T, f func(t *testing.T)) {
	ks := yencsimd.Kernels()
	t.Cleanup(func() { yencsimd.SetKernel(ks[0]) })
	for _, k := range ks {
		if err := yencsimd.SetKernel(k); err != nil {
			t.Fatal(err)
		}
		t.Run(k, f)
	}
}

// TestFetch fetches every post of every group over one connection, with textproto's default
// 4 KiB buffer and with the 64 KiB the README recommends.
func TestFetch(t *testing.T) {
	s := fakeserver.New(t, fakeserver.Options{Store: newStore()})
	eachKernel(t, func(t *testing.T) {
		for _, bufSize := range []int{4 << 10, 64 << 10} {
			c := mustDial(t, s.Addr(), bufSize)
			for _, g := range groups {
				c.group(t, g.name)
				for n := int64(1); n <= int64(len(sizes)); n++ {
					data, m, err := c.body(n)
					if err := check(data, m, err, g.p, n, sizeOf(n)); err != nil {
						t.Fatalf("buffer %d, %s: %v", bufSize, g.name, err)
					}
				}
			}
		}
	})
}

// article wraps body lines into a stored post.
func article(id string, body []string) fakeserver.Article {
	mid := "<" + id + "@e2e.invalid>"
	return fakeserver.Article{
		MessageID: mid,
		Headers:   []string{"From: poster@e2e.invalid", "Subject: " + id, "Message-ID: " + mid},
		Body:      body,
	}
}

// TestMultipart fetches the parts of three files and puts each file together from Meta.Begin,
// as the README shows.
func TestMultipart(t *testing.T) {
	files := []struct {
		name       string
		p          yenc.Profile
		size, part int
		o          yenc.Options
	}{
		{"a.bin", yenc.Random, 1_000_000, 384_000, yenc.Options{}},
		{"b.bin", yenc.Edges, 100_001, 25_000, yenc.Options{RawDots: true, RawSpaces: true}}, // a last part of 1 byte
		{"c.bin", yenc.Escapes, 65_536, 16_384, yenc.Options{Line: 64}},
	}
	type post struct {
		n          int64
		file, part int
	}
	st := fakeserver.NewStore()
	st.AddGroup("yenc.parts", "multi-part posts")
	var posts []post
	datas := make([][]byte, len(files))
	totals := make([]int, len(files))
	for i, f := range files {
		datas[i] = yenc.Data(f.p, f.size, uint64(100+i))
		parts := yenc.EncodeParts(f.name, datas[i], f.part, f.o)
		totals[i] = len(parts)
		for k, lines := range parts {
			n := st.AddArticle("yenc.parts", article(fmt.Sprintf("%s.%d", f.name, k+1), lines))
			posts = append(posts, post{n, i, k + 1})
		}
	}
	s := fakeserver.New(t, fakeserver.Options{Store: st})
	eachKernel(t, func(t *testing.T) {
		c := mustDial(t, s.Addr(), 64<<10)
		c.group(t, "yenc.parts")
		got := make([][]byte, len(files))
		for i, f := range files {
			got[i] = make([]byte, f.size)
		}
		for _, p := range posts {
			f, whole := files[p.file], datas[p.file]
			data, m, err := c.body(p.n)
			if err != nil {
				t.Fatalf("%s part %d: %v", f.name, p.part, err)
			}
			begin := (p.part-1)*f.part + 1
			end := min(begin+f.part-1, f.size)
			if m.Name != f.name || m.Size != int64(f.size) || m.Part != p.part || m.Total != totals[p.file] ||
				m.Begin != int64(begin) || m.End != int64(end) || !m.HasCRC || m.CRC32 != crc32.ChecksumIEEE(whole[begin-1:end]) {
				t.Fatalf("%s part %d: meta %+v", f.name, p.part, m)
			}
			copy(got[p.file][m.Begin-1:], data)
		}
		for i, f := range files {
			if !bytes.Equal(got[i], datas[i]) {
				t.Errorf("%s: the parts don't add up (first difference at byte %d)", f.name, firstDiff(got[i], datas[i]))
			}
		}
	})
}

// TestBroken fetches broken posts: each must give its error and leave the connection ready
// for the next command, which a good post fetched after each one checks.
func TestBroken(t *testing.T) {
	data := yenc.Data(yenc.Random, 5000, 1)
	parts := yenc.EncodeParts("p.bin", data, 2000, yenc.Options{BadCRC: true})
	broken := []struct {
		name string
		body []string
		want error
	}{
		{"no =yend", yenc.Encode("x.bin", data, yenc.Options{NoEnd: true}), yencsimd.ErrTruncated},
		{"no =ybegin", yenc.Encode("x.bin", data, yenc.Options{NoBegin: true}), yencsimd.ErrNoYbegin},
		{"size", yenc.Encode("x.bin", data, yenc.Options{BadSize: true}), yencsimd.ErrSize},
		{"crc32", yenc.Encode("x.bin", data, yenc.Options{BadCRC: true}), yencsimd.ErrCRC},
		{"pcrc32", parts[1], yencsimd.ErrCRC},
		{"not yEnc", []string{"hello", ".a dot-stuffed line", "world"}, yencsimd.ErrNoYbegin},
		{"empty", nil, yencsimd.ErrNoYbegin},
	}
	st := fakeserver.NewStore()
	st.AddGroup("yenc.broken", "broken posts")
	good := st.AddArticle("yenc.broken", article("good", yenc.Encode("good.bin", data, yenc.Options{})))
	nums := make([]int64, len(broken))
	for i, b := range broken {
		nums[i] = st.AddArticle("yenc.broken", article(fmt.Sprintf("broken%d", i), b.body))
	}
	s := fakeserver.New(t, fakeserver.Options{Store: st})
	eachKernel(t, func(t *testing.T) {
		c := mustDial(t, s.Addr(), 4<<10)
		c.group(t, "yenc.broken")
		for i, b := range broken {
			if _, _, err := c.body(nums[i]); !errors.Is(err, b.want) {
				t.Errorf("%s: %v, want %v", b.name, err, b.want)
			}
			got, _, err := c.body(good)
			if err != nil || !bytes.Equal(got, data) {
				t.Fatalf("the good post after %q: %d bytes, %v", b.name, len(got), err)
			}
		}
	})
}

// TestCutOff: the server closes the connection in the middle of a 750 KB post.
func TestCutOff(t *testing.T) {
	s := fakeserver.New(t, fakeserver.Options{
		Store:  newStore(),
		Faults: fakeserver.Faults{DropInMultiline: map[string]int{"BODY": 50}},
	})
	eachKernel(t, func(t *testing.T) {
		c := mustDial(t, s.Addr(), 4<<10)
		c.group(t, "yenc.random")
		data, _, err := c.body(int64(len(sizes))) // the status line, =ybegin and 48 of ~5900 data lines
		if !errors.Is(err, yencsimd.ErrTruncated) {
			t.Fatalf("%d bytes, %v; want ErrTruncated", len(data), err)
		}
	})
}

// TestSlow: the server writes posts one byte at a time (Drip) or one line at a time
// (LineDelay), so Reader gets them in the smallest pieces.
func TestSlow(t *testing.T) {
	t.Run("byte at a time", func(t *testing.T) {
		small := func(n int64) int { return int(n) * 37 } // 37 to 296 bytes
		st := fakeserver.NewStore()
		st.AddGenerated("yenc.edges.raw", "", yenc.Generator(yenc.Edges, 1, 8, small, yenc.Options{RawDots: true, RawSpaces: true}))
		st.AddGenerated("yenc.escapes", "", yenc.Generator(yenc.Escapes, 1, 8, small, yenc.Options{}))
		s := fakeserver.New(t, fakeserver.Options{
			Store:  st,
			Faults: fakeserver.Faults{Drip: map[string]time.Duration{"BODY": time.Microsecond}},
		})
		// A post that arrives a byte at a time never fills a vector, so one kernel is enough.
		c := mustDial(t, s.Addr(), 4<<10)
		for _, g := range []struct {
			name string
			p    yenc.Profile
		}{{"yenc.edges.raw", yenc.Edges}, {"yenc.escapes", yenc.Escapes}} {
			c.group(t, g.name)
			for n := int64(1); n <= 8; n++ {
				data, m, err := c.body(n)
				if err := check(data, m, err, g.p, n, small(n)); err != nil {
					t.Fatalf("%s: %v", g.name, err)
				}
			}
		}
	})
	t.Run("line at a time", func(t *testing.T) {
		s := fakeserver.New(t, fakeserver.Options{
			Store:  newStore(),
			Faults: fakeserver.Faults{LineDelay: map[string]time.Duration{"BODY": 50 * time.Microsecond}},
		})
		eachKernel(t, func(t *testing.T) {
			c := mustDial(t, s.Addr(), 64<<10)
			for _, g := range groups {
				c.group(t, g.name)
				for n := int64(18); n <= 22; n++ { // 1000 to 65536 bytes
					data, m, err := c.body(n)
					if err := check(data, m, err, g.p, n, sizeOf(n)); err != nil {
						t.Fatalf("%s: %v", g.name, err)
					}
				}
			}
		})
	})
}

// TestParallel: 8 connections fetch 750 KB posts at the same time.
func TestParallel(t *testing.T) {
	const conns, posts, each = 8, 24, 12
	st := fakeserver.NewStore()
	st.AddGenerated("yenc.big", "750 KB posts", yenc.Generator(yenc.Random, 1, posts, nil, yenc.Options{}))
	s := fakeserver.New(t, fakeserver.Options{Store: st})
	eachKernel(t, func(t *testing.T) {
		var wg sync.WaitGroup
		errs := make(chan error, conns)
		for g := range conns {
			wg.Add(1)
			go func() {
				defer wg.Done()
				c, err := dial(s.Addr(), 64<<10)
				if err != nil {
					errs <- err
					return
				}
				defer c.nc.Close()
				if err := c.cmd(211, "GROUP yenc.big"); err != nil {
					errs <- err
					return
				}
				for i := range each {
					n := int64((g*5+i)%posts + 1)
					data, m, err := c.body(n)
					if err := check(data, m, err, yenc.Random, n, 750_000); err != nil {
						errs <- fmt.Errorf("connection %d: %w", g, err)
						return
					}
				}
			}()
		}
		wg.Wait()
		close(errs)
		for err := range errs {
			t.Error(err)
		}
	})
}

// wire returns body lines as a server sends them: dot-stuffed, with CRLF and the closing ".".
func wire(lines []string) []byte {
	var b bytes.Buffer
	for _, l := range lines {
		if strings.HasPrefix(l, ".") {
			b.WriteByte('.')
		}
		b.WriteString(l)
		b.WriteString("\r\n")
	}
	b.WriteString(".\r\n")
	return b.Bytes()
}

// TestEncoderAgainstRapidyenc decodes the encoder's posts with rapidyenc as well: its C
// library when built with cgo, else its Go port. Without that check, a mistake shared by the
// encoder and Reader would pass every test above.
func TestEncoderAgainstRapidyenc(t *testing.T) {
	t.Logf("rapidyenc decodes with %s", rapidyenc.DecodeKernel())
	decode := func(body []string) ([]byte, error) {
		resp, err := rapidyenc.NewDecoder(bytes.NewReader(wire(body)), rapidyenc.WithStatusLineAlreadyRead()).Next()
		if resp == nil {
			return nil, err
		}
		return resp.Data, err
	}
	for _, g := range groups {
		for n := int64(1); n <= int64(len(sizes)); n++ {
			want := yenc.Data(g.p, sizeOf(n), uint64(n))
			got, err := decode(yenc.Encode("x.bin", want, g.o))
			if err != nil || !bytes.Equal(got, want) {
				t.Fatalf("%s, %d bytes: rapidyenc gave %d bytes, %v", g.name, len(want), len(got), err)
			}
		}
	}
	data := yenc.Data(yenc.Edges, 100_001, 7)
	for k, body := range yenc.EncodeParts("p.bin", data, 25_000, yenc.Options{RawDots: true, RawSpaces: true}) {
		want := data[k*25_000 : min((k+1)*25_000, len(data))]
		if got, err := decode(body); err != nil || !bytes.Equal(got, want) {
			t.Fatalf("part %d: rapidyenc gave %d bytes, %v", k+1, len(got), err)
		}
	}
}
