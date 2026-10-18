//go:build goexperiment.simd && amd64

package yencsimd

// cpuKernels lists the decode paths this CPU can run, fastest first: the
// conditions decode() dispatches on, taken before any test changes them.
var cpuKernels = func() []string {
	var k []string
	if hasVBMI2 {
		k = append(k, "avx512vbmi2")
	}
	if hasAVX2 {
		k = append(k, "avx2")
	}
	if hasAVX {
		k = append(k, "avx")
	}
	return append(k, "scalar")
}()

func kernels() []string { return cpuKernels }

// withKernel runs f with the decoder limited to the named path.
func withKernel(name string, f func()) {
	v, a2, a := hasVBMI2, hasAVX2, hasAVX
	defer func() { hasVBMI2, hasAVX2, hasAVX = v, a2, a }()
	hasVBMI2 = name == "avx512vbmi2" && v
	hasAVX2 = (name == "avx512vbmi2" || name == "avx2") && a2
	hasAVX = name != "scalar" && a
	f()
}
