package yencsimd

import (
	"slices"
	"testing"
)

// kernels lists the decode paths this CPU can run, fastest first.
func kernels() []string { return Kernels() }

// withKernel runs f with the decoder limited to the named path, then puts
// back the one that was in use.
func withKernel(name string, f func()) {
	prev := Kernel()
	defer SetKernel(prev)
	if err := SetKernel(name); err != nil {
		panic(err)
	}
	f()
}

// TestKernels checks Kernels, Kernel and SetKernel on this build and CPU:
// Kernels lists distinct paths ending with "scalar" and starting with the
// one in use; SetKernel takes each of them (Kernel then names it) and
// refuses an unknown name without changing the kernel.
func TestKernels(t *testing.T) {
	all := Kernels()
	if len(all) == 0 || all[len(all)-1] != "scalar" {
		t.Fatalf("Kernels() = %q, want a list ending with scalar", all)
	}
	if len(slices.Compact(slices.Sorted(slices.Values(all)))) != len(all) {
		t.Fatalf("Kernels() = %q has a name twice", all)
	}
	if k := Kernel(); k != all[0] {
		t.Fatalf("Kernel() = %q, want the fastest, %q (did a test leave another set?)", k, all[0])
	}
	defer SetKernel(all[0])
	for _, k := range all {
		if err := SetKernel(k); err != nil {
			t.Fatalf("SetKernel(%q): %v", k, err)
		}
		if got := Kernel(); got != k {
			t.Fatalf("after SetKernel(%q), Kernel() = %q", k, got)
		}
		if err := SetKernel("no-such-kernel"); err == nil {
			t.Fatal("SetKernel(no-such-kernel): no error")
		}
		if got := Kernel(); got != k {
			t.Fatalf("a refused SetKernel changed the kernel from %q to %q", k, got)
		}
		withKernel("scalar", func() {
			if got := Kernel(); got != "scalar" {
				t.Fatalf("inside withKernel(scalar), Kernel() = %q", got)
			}
		})
		if got := Kernel(); got != k {
			t.Fatalf("withKernel did not put back %q: Kernel() = %q", k, got)
		}
	}
	if !slices.Equal(Kernels(), all) {
		t.Fatalf("Kernels() changed with SetKernel: %q, was %q", Kernels(), all)
	}
}
