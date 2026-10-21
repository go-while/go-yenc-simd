//go:build goexperiment.simd && arm64

package yencsimd

func kernels() []string { return []string{"neon", "scalar"} }

// withKernel runs f with the decoder limited to the named path.
func withKernel(name string, f func()) {
	n := hasNEON
	defer func() { hasNEON = n }()
	hasNEON = name == "neon"
	f()
}
