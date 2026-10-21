//go:build !(goexperiment.simd && (amd64 || arm64))

package yencsimd

import "errors"

// Kernel reports which fast path the decoder uses.
func Kernel() string { return "scalar" }

// Kernels lists the decode paths this build can run: only the scalar one.
func Kernels() []string { return []string{"scalar"} }

// SetKernel limits the decoder to the named path, one of Kernels().
func SetKernel(name string) error {
	if name != "scalar" {
		return errors.New("yencsimd: kernel " + name + " is not available (needs GOEXPERIMENT=simd on amd64 or arm64)")
	}
	return nil
}

func decode(dst, src []byte, state *State) (int, int, End) { return decodeScalar(dst, src, state) }
