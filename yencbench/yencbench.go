// Package yencbench measures how fast yencsimd decodes on this machine, with
// the same inputs as the package's own benchmarks. Programs can call Run to
// print a table (for example behind a command-line flag) or Measure to get
// the numbers; cmd/yencbench is the command-line version.
package yencbench

import (
	"bytes"
	"fmt"
	"hash/crc32"
	"io"
	"math/rand/v2"
	"os"
	"runtime"
	"strings"
	"time"

	yencsimd "github.com/go-while/go-yenc-simd"
)

// Options control a run. The zero value measures every kernel this CPU can
// run, 1 s per measurement, with 750 KB articles.
type Options struct {
	Time    time.Duration // per measurement; 1 s if zero
	Size    int           // data bytes per article; 750000 if zero
	Kernels []string      // kernels to measure (see yencsimd.Kernels); all if nil
}

// Result is one measurement: MBps is bytes of input per second, in millions.
type Result struct {
	Kernel     string
	Input      string // random128, plain, escapes, or article
	MBps       float64
	Iterations int
}

// Inputs are the articles of the package's benchmarks: random128 is random
// data (an escape every ~64 bytes, 128-byte lines), plain is data that needs
// no escapes, escapes has an escape in every second byte. The article
// measurement runs Decode (header parsing, size and CRC32 check) on the
// random128 article; the others run DecodeIncremental on the body.
var Inputs = []string{"random128", "plain", "escapes", "article"}

// Measure runs the benchmark and returns a result per kernel and input, in
// Inputs order. The kernel in use is restored afterwards.
func Measure(o Options) ([]Result, error) {
	if o.Time <= 0 {
		o.Time = time.Second
	}
	if o.Size <= 0 {
		o.Size = 750_000
	}
	kernels := o.Kernels
	if kernels == nil {
		kernels = yencsimd.Kernels()
	}
	defer yencsimd.SetKernel(yencsimd.Kernels()[0])

	random := article(o.Size, 7, func(r *rand.Rand, i int) byte { return byte(r.Uint32()) })
	plain := article(o.Size, 9, func(r *rand.Rand, i int) byte { return plainByte(r) })
	escapes := article(o.Size, 10, func(r *rand.Rand, i int) byte {
		if i%2 == 1 {
			return []byte{214, 224, 227, 19}[r.IntN(4)] // encode to NUL, \n, \r, '='
		}
		return plainByte(r)
	})
	body := func(art []byte) []byte { return art[bytes.Index(art, []byte("\r\n"))+2:] }

	var results []Result
	for _, k := range kernels {
		if err := yencsimd.SetKernel(k); err != nil {
			return results, err
		}
		for _, in := range []struct {
			name string
			art  []byte
		}{{"random128", random}, {"plain", plain}, {"escapes", escapes}, {"article", random}} {
			var once func() error
			var n int
			if in.name == "article" {
				n = len(in.art)
				once = func() error {
					data, _, err := yencsimd.Decode(in.art)
					if err != nil || len(data) != o.Size {
						return fmt.Errorf("yencbench: %s/%s: Decode gave %d bytes, %v", k, in.name, len(data), err)
					}
					return nil
				}
			} else {
				src := body(in.art)
				n = len(src)
				dst := make([]byte, len(src))
				once = func() error {
					st := yencsimd.StateCRLF
					nd, _, end, err := yencsimd.DecodeIncremental(dst, src, &st)
					if err != nil || end != yencsimd.EndControl || nd != o.Size {
						return fmt.Errorf("yencbench: %s/%s: DecodeIncremental gave %d bytes, end %d, %v", k, in.name, nd, end, err)
					}
					return nil
				}
			}
			if err := once(); err != nil {
				return results, err
			}
			start := time.Now()
			iters := 0
			var elapsed time.Duration
			for elapsed < o.Time {
				if err := once(); err != nil {
					return results, err
				}
				iters++
				elapsed = time.Since(start)
			}
			results = append(results, Result{k, in.name, float64(n) * float64(iters) / elapsed.Seconds() / 1e6, iters})
		}
	}
	return results, nil
}

// Run measures and prints a table: inputs as rows, kernels as columns.
func Run(w io.Writer, o Options) error {
	results, err := Measure(o)
	if err != nil {
		return err
	}
	size := o.Size
	if size <= 0 {
		size = 750_000
	}
	var kernels []string
	mbps := map[string]map[string]float64{}
	for _, r := range results {
		if mbps[r.Kernel] == nil {
			kernels = append(kernels, r.Kernel)
			mbps[r.Kernel] = map[string]float64{}
		}
		mbps[r.Kernel][r.Input] = r.MBps
	}
	cpu := cpuName()
	if cpu != "" {
		cpu = ", " + cpu
	}
	fmt.Fprintf(w, "go-yenc-simd: %s %s/%s%s; decoding uses %s\n", runtime.Version(), runtime.GOOS, runtime.GOARCH, cpu, yencsimd.Kernel())
	fmt.Fprintf(w, "%-30s", fmt.Sprintf("%d KB article, MB/s", size/1000))
	for _, k := range kernels {
		fmt.Fprintf(w, " %12s", k)
	}
	fmt.Fprintln(w)
	for _, in := range Inputs {
		label := "DecodeIncremental " + in
		if in == "article" {
			label = "Decode (article, CRC32)"
		}
		fmt.Fprintf(w, "%-30s", label)
		for _, k := range kernels {
			fmt.Fprintf(w, " %12.0f", mbps[k][in])
		}
		fmt.Fprintln(w)
	}
	return nil
}

// article yEnc-encodes size bytes from gen (seeded) with 128-byte lines, the
// critical bytes (and tab, space and dot where yEnc wants it) escaped, and a
// crc32= in the =yend line, like the package's bench_test.go.
func article(size int, seed uint64, gen func(r *rand.Rand, i int) byte) []byte {
	r := rand.New(rand.NewPCG(seed, seed+1))
	data := make([]byte, size)
	for i := range data {
		data[i] = gen(r, i)
	}
	var b bytes.Buffer
	fmt.Fprintf(&b, "=ybegin line=128 size=%d name=bench.bin\r\n", len(data))
	col := 0
	for i, c := range data {
		o := c + 42
		last := i == len(data)-1 || col >= 127
		esc := o == 0 || o == '\n' || o == '\r' || o == '=' ||
			(o == '\t' || o == ' ') && (col == 0 || last) ||
			o == '.' && col == 0
		if esc {
			b.WriteByte('=')
			b.WriteByte(o + 64)
			col += 2
		} else {
			b.WriteByte(o)
			col++
		}
		if col >= 128 && i != len(data)-1 {
			b.WriteString("\r\n")
			col = 0
		}
	}
	fmt.Fprintf(&b, "\r\n=yend size=%d crc32=%08x\r\n.\r\n", len(data), crc32.ChecksumIEEE(data))
	return b.Bytes()
}

// plainByte returns a data byte that encodes to a plain byte: not '=', '\r',
// '\n', NUL, '.', tab or space (the last three are escaped at line starts).
func plainByte(r *rand.Rand) byte {
	for {
		c := byte(r.Uint32())
		switch c + 42 {
		case '=', '\r', '\n', 0, '.', '\t', ' ':
			continue
		}
		return c
	}
}

// cpuName is the CPU model from /proc/cpuinfo, or "".
func cpuName() string {
	b, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(line, "model name") {
			if _, v, ok := strings.Cut(line, ":"); ok {
				return strings.Join(strings.Fields(v), " ")
			}
		}
	}
	return ""
}
