package yencbench

import (
	"bytes"
	"strings"
	"testing"
	"time"

	yencsimd "github.com/go-while/go-yenc-simd"
)

// TestMeasure is a smoke test: a short run over every kernel gives a result
// per kernel and input, and the kernel the caller had set is still set
// afterwards (here the slowest one, not the fastest that Measure used to
// switch to), also when Measure fails on an unknown kernel.
func TestMeasure(t *testing.T) {
	all := yencsimd.Kernels()
	last := all[len(all)-1]
	prev := yencsimd.Kernel()
	defer yencsimd.SetKernel(prev)
	if err := yencsimd.SetKernel(last); err != nil {
		t.Fatal(err)
	}

	o := Options{Time: time.Millisecond, Size: 10_000}
	results, err := Measure(o)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != len(all)*len(Inputs) {
		t.Fatalf("%d results, want %d (%d kernels x %d inputs)", len(results), len(all)*len(Inputs), len(all), len(Inputs))
	}
	for i, r := range results {
		if r.Kernel != all[i/len(Inputs)] || r.Input != Inputs[i%len(Inputs)] || r.Iterations < 1 || r.MBps <= 0 {
			t.Fatalf("result %d: %+v", i, r)
		}
	}
	if k := yencsimd.Kernel(); k != last {
		t.Fatalf("after Measure the kernel is %q, want %q (set before)", k, last)
	}

	o.Kernels = []string{"no-such-kernel"}
	if _, err := Measure(o); err == nil {
		t.Fatal("Measure with an unknown kernel: no error")
	}
	if k := yencsimd.Kernel(); k != last {
		t.Fatalf("after a failed Measure the kernel is %q, want %q", k, last)
	}

	var out bytes.Buffer
	if err := Run(&out, Options{Time: time.Millisecond, Size: 10_000, Kernels: []string{last}}); err != nil {
		t.Fatal(err)
	}
	if s := out.String(); !strings.Contains(s, "decoding uses "+last) || !strings.Contains(s, "Decode (article, CRC32)") {
		t.Fatalf("Run printed:\n%s", s)
	}
}
