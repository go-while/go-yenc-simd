package yencsimd

import "testing"

// TestBoundaries puts every string of up to 4 of the bytes that matter at
// every offset of plain data, with tails of several lengths, so each one
// straddles every vector block edge and every loop entry and exit condition,
// and compares each kernel with the state machine. Bugs of this class (a
// "\r\n." or an escape split by a block edge) were found in other ports of
// rapidyenc's kernels.
func TestBoundaries(t *testing.T) {
	var pats [][]byte
	var gen func(p []byte)
	gen = func(p []byte) {
		if len(p) > 0 {
			pats = append(pats, append([]byte(nil), p...))
		}
		if len(p) < 4 {
			for _, c := range []byte("=\r\n.ya") {
				gen(append(p, c))
			}
		}
	}
	gen(nil)
	plain := make([]byte, 300)
	for i := range plain {
		plain[i] = 'A' + byte(i%26)
	}
	for _, k := range kernels() {
		if k == "scalar" {
			continue
		}
		n := 0
		for _, pat := range pats {
			for off := 0; off <= 140; off++ {
				for _, tail := range []int{0, 3, 24, 47, 70, 140} {
					src := append(append(append([]byte(nil), plain[:off]...), pat...), plain[off:off+tail]...)
					states := []State{StateNone, StateCRLF}
					if off < 4 {
						states = []State{0, 1, 2, 3, 4, 5, 6}
					}
					for _, st := range states {
						compareWithScalar(t, k, src, st)
						n++
					}
				}
			}
		}
		t.Logf("kernel %s: %d inputs", k, n)
	}
}
