//go:build goexperiment.simd && amd64

package yencsimd

import "simd/archsimd"

// kernels lists the decode paths this CPU can run, fastest first.
func kernels() []string {
	var k []string
	if archsimd.X86.AVX512VBMI2() {
		k = append(k, "avx512vbmi2")
	}
	if archsimd.X86.AVX2() {
		k = append(k, "avx2")
	}
	if archsimd.X86.AVX() {
		k = append(k, "avx")
	}
	return append(k, "scalar")
}

// withKernel runs f with the decoder limited to the named path.
func withKernel(name string, f func()) {
	v, a2, a := hasVBMI2, hasAVX2, hasAVX
	defer func() { hasVBMI2, hasAVX2, hasAVX = v, a2, a }()
	hasVBMI2 = name == "avx512vbmi2" && v
	hasAVX2 = (name == "avx512vbmi2" || name == "avx2") && a2
	hasAVX = name != "scalar" && a
	f()
}
