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

// benchInputs are articles of ~750 KB of data that stress different paths:
// random128 is benchBody (an escape every ~60 bytes, a line break every
// 130), text encodes to plain bytes only (line breaks are the only
// specials), escapes has an escape in every second byte.
var benchInputs = []benchInput{
	{"random128", benchBody, benchSize},
	benchInputFrom("text", 9, func(r *rand.Rand, i int) byte { return benchPlain(r) }),
	benchInputFrom("escapes", 10, func(r *rand.Rand, i int) byte {
		if i%2 == 1 {
			return []byte{214, 224, 227, 19}[r.IntN(4)] // encode to NUL, \n, \r, '='
		}
		return benchPlain(r)
	}),
}

type benchInput struct {
	name string
	body []byte
	size int
}

// benchPlain returns a data byte that encodes to a plain byte: not '=', '\r',
// '\n', NUL, '.', tab or space (the last three are escaped at line starts).
func benchPlain(r *rand.Rand) byte {
	for {
		c := byte(r.Uint32())
		switch c + 42 {
		case '=', '\r', '\n', 0, '.', '\t', ' ':
			continue
		}
		return c
	}
}

func benchInputFrom(name string, seed uint64, gen func(r *rand.Rand, i int) byte) benchInput {
	r := rand.New(rand.NewPCG(seed, seed+1))
	data := make([]byte, 750_000)
	for i := range data {
		data[i] = gen(r, i)
	}
	art := encode(data, true)
	return benchInput{name, art[bytes.Index(art, []byte("\r\n"))+2:], len(data)}
}

func BenchmarkDecode(b *testing.B) {
	for _, in := range benchInputs {
		b.Run(in.name, func(b *testing.B) {
			for _, k := range kernels() {
				b.Run(k, func(b *testing.B) {
					withKernel(k, func() {
						dst := make([]byte, len(in.body))
						b.SetBytes(int64(len(in.body)))
						for b.Loop() {
							st := StateCRLF
							nd, _, end, _ := DecodeIncremental(dst, in.body, &st)
							if end != EndControl || nd != in.size {
								b.Fatalf("end=%d nd=%d", end, nd)
							}
						}
					})
				})
			}
		})
	}
}
