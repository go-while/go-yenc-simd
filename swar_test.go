package yencsimd

import (
	"encoding/binary"
	"math/bits"
	"math/rand/v2"
	"testing"
)

// TestWord checks word against the byte-wise definition: every byte value in
// every lane, with the other lanes random (so a borrow or a carry from a
// neighbour would show), as the only special byte and with specials around.
func TestWord(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	var in, out [8]byte
	for lane := range 8 {
		for c := range 256 {
			for iter := range 256 {
				binary.LittleEndian.PutUint64(in[:], r.Uint64())
				switch {
				case iter == 0:
					in = [8]byte{}
				case iter == 1:
					in = [8]byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}
				case iter < 128: // no other special byte
					for k, b := range in {
						if special[b] {
							in[k] = b + 1
						}
					}
				}
				in[lane] = byte(c)
				first := 8 // the first special byte
				for k, b := range in {
					if special[b] {
						first = k
						break
					}
				}
				o, tt := word(binary.LittleEndian.Uint64(in[:]), ones, highs, eqs, crs, lfs, k42s)
				binary.LittleEndian.PutUint64(out[:], o)
				for k, b := range in {
					if out[k] != b-42 {
						t.Fatalf("word(% x): byte %d = %#x, want %#x", in, k, out[k], b-42)
					}
				}
				if got := bits.TrailingZeros64(tt) >> 3; got != first || tt != 0 && tt&(1<<(8*got+7)) == 0 {
					t.Fatalf("word(% x): t = %#x, first special byte %d, want %d", in, tt, got, first)
				}
			}
		}
	}
}
