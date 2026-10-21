package yencsimd

import (
	"bytes"
	"math/rand/v2"
	"testing"
)

// The vector paths are checked against the state machine (decodeScalar), which
// the golden table ties to rapidyenc's C decoder, and the scalar path, whose
// word loop (decodePlain) skips the state machine for plain bytes, escapes
// and line breaks, against the state machine alone (decodeSteps).

// randomRaw makes input dense with the bytes that matter (=, \r, \n, ., y),
// with plain runs in between, so every vector rule and block boundary is hit.
func randomRaw(r *rand.Rand, n int) []byte {
	alphabet := []byte("=\r\n.y=\r\naZ\x00\xff\t ")
	b := make([]byte, n)
	for i := 0; i < n; {
		if r.IntN(4) == 0 {
			run := min(n-i, 1+r.IntN(80))
			for j := 0; j < run; j++ {
				c := byte(r.IntN(256))
				for special[c] {
					c = byte(r.IntN(256))
				}
				b[i+j] = c
			}
			i += run
			continue
		}
		b[i] = alphabet[r.IntN(len(alphabet))]
		i++
	}
	return b
}

// decodeSteps is the state machine alone, one byte at a time.
func decodeSteps(dst, src []byte, state *State) (int, int, End) {
	st, p := *state, 0
	for i, c := range src {
		var end End
		st, p, end = step(st, c, dst, p)
		if end != EndNone {
			*state = st
			return p, i + 1, end
		}
	}
	*state = st
	return p, len(src), EndNone
}

// compareWithScalar decodes src with the kernel and with decodeScalar (with
// decodeSteps if the kernel is "scalar") and fails if they differ.
func compareWithScalar(t testing.TB, kernel string, src []byte, st State) {
	t.Helper()
	ref := decodeScalar
	if kernel == "scalar" {
		ref = decodeSteps
	}
	sDst := make([]byte, len(src))
	sSt := st
	snd, sns, send := ref(sDst, src, &sSt)
	var vnd, vns int
	var vend End
	vDst := make([]byte, len(src))
	vSt := st
	withKernel(kernel, func() { vnd, vns, vend, _ = DecodeIncremental(vDst, src, &vSt) })
	if vnd != snd || vns != sns || vend != send || vSt != sSt || !bytes.Equal(vDst[:vnd], sDst[:snd]) {
		t.Fatalf("kernel %s differs from the state machine, state %d, src %q\n %s: nDst=%d nSrc=%d end=%d state=%d\n scalar: nDst=%d nSrc=%d end=%d state=%d",
			kernel, st, src, kernel, vnd, vns, vend, vSt, snd, sns, send, sSt)
	}
}

func TestVectorPathsMatchStateMachine(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	for _, k := range kernels() {
		for iter := 0; iter < 50000; iter++ {
			n := r.IntN(300)
			if iter%10 == 0 {
				n = r.IntN(4000)
			}
			compareWithScalar(t, k, randomRaw(r, n), State(r.IntN(7)))
		}
	}
}

func FuzzVectorPathsMatchStateMachine(f *testing.F) {
	f.Add([]byte("abc\r\n..def\r\n=yend"), uint8(0))
	f.Add([]byte("=\r\n.\r\n"), uint8(1))
	f.Add([]byte("\r\n.=yx"), uint8(4))
	f.Fuzz(func(t *testing.T, src []byte, st uint8) {
		for _, k := range kernels() {
			compareWithScalar(t, k, src, State(st%7))
		}
	})
}
