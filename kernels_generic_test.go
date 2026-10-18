//go:build !(goexperiment.simd && amd64)

package yencsimd

func kernels() []string                { return []string{"scalar"} }
func withKernel(name string, f func()) { f() }
