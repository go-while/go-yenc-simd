package yencsimd

import (
	"bytes"
	"math/rand/v2"
	"testing"
)

// benchBody: ~750 KB of random data, yEnc encoded (128-byte lines) and
// dot-stuffed, up to and including "=yend".
var benchBody, benchSize = func() ([]byte, int) {
	r := rand.New(rand.NewPCG(7, 8))
	data := make([]byte, 750_000)
	for i := range data {
		data[i] = byte(r.Uint32())
	}
	art := encode(data, true)
	return art[bytes.Index(art, []byte("\r\n"))+2:], len(data)
}()

func BenchmarkDecode(b *testing.B) {
	for _, k := range kernels() {
		b.Run(k, func(b *testing.B) {
			withKernel(k, func() {
				dst := make([]byte, len(benchBody))
				b.SetBytes(int64(len(benchBody)))
				for b.Loop() {
					st := StateCRLF
					nd, _, end, _ := DecodeIncremental(dst, benchBody, &st)
					if end != EndControl || nd != benchSize {
						b.Fatalf("end=%d nd=%d", end, nd)
					}
				}
			})
		})
	}
}
