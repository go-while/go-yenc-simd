package yencsimd

import (
	"bytes"
	"math/rand/v2"
	"strconv"
	"testing"
)

// encode yEnc-encodes data with 128-byte lines and dot-stuffs it for NNTP. The
// critical bytes (NUL, LF, CR, '=') are always escaped, tab and space at line
// starts and ends too; a dot at a line start is escaped only if escapeDot,
// otherwise it is dot-stuffed ("..") as NNTP requires.
func encode(data []byte, escapeDot bool) []byte {
	var b bytes.Buffer
	b.WriteString("=ybegin line=128 size=" + strconv.Itoa(len(data)) + " name=test.bin\r\n")
	col := 0
	for i, c := range data {
		o := c + 42
		last := i == len(data)-1 || col >= 127
		esc := o == 0 || o == '\n' || o == '\r' || o == '=' ||
			(o == '\t' || o == ' ') && (col == 0 || last) ||
			o == '.' && col == 0 && escapeDot
		switch {
		case esc:
			b.WriteByte('=')
			b.WriteByte(o + 64)
			col += 2
		case o == '.' && col == 0:
			b.WriteString("..")
			col++
		default:
			b.WriteByte(o)
			col++
		}
		if col >= 128 && i != len(data)-1 {
			b.WriteString("\r\n")
			col = 0
		}
	}
	b.WriteString("\r\n=yend size=" + strconv.Itoa(len(data)) + "\r\n.\r\n")
	return b.Bytes()
}

// decodeChunks decodes src in pieces cut at the given offsets, like a reader would.
func decodeChunks(src []byte, cuts []int) ([]byte, int, End) {
	out := make([]byte, 0, len(src))
	dst := make([]byte, len(src))
	st := StateCRLF
	pos := 0
	for _, cut := range append(cuts, len(src)) {
		if cut <= pos {
			continue
		}
		nd, ns, end, err := DecodeIncremental(dst, src[pos:cut], &st)
		if err != nil {
			panic(err)
		}
		out = append(out, dst[:nd]...)
		pos += ns
		if end != EndNone {
			return out, pos, end
		}
	}
	return out, pos, EndNone
}

func randomCuts(r *rand.Rand, n int) []int {
	cuts := make([]int, r.IntN(8))
	for i := range cuts {
		cuts[i] = r.IntN(n + 1)
	}
	for i := 1; i < len(cuts); i++ {
		for j := i; j > 0 && cuts[j] < cuts[j-1]; j-- {
			cuts[j], cuts[j-1] = cuts[j-1], cuts[j]
		}
	}
	return cuts
}

func TestRoundTrip(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, k := range kernels() {
		withKernel(k, func() {
			for iter := 0; iter < 200; iter++ {
				data := make([]byte, r.IntN(300_000))
				for i := range data {
					data[i] = byte(r.Uint32())
				}
				if iter%3 == 0 { // escape-heavy: bytes that encode to critical characters
					for i := range data {
						data[i] = []byte{214, 224, 227, 19, 4, 223, 246, 0}[r.IntN(8)]
					}
				}
				art := encode(data, iter%2 == 0)
				body := art[bytes.Index(art, []byte("\r\n"))+2:]
				out, pos, end := decodeChunks(body, randomCuts(r, len(body)))
				if end != EndControl || !bytes.Equal(out, data) {
					t.Fatalf("kernel %s, article %d (%d bytes): end=%d, %d bytes back, equal=%v", k, iter, len(data), end, len(out), bytes.Equal(out, data))
				}
				if !bytes.HasPrefix(body[pos-2:], []byte("=yend")) {
					t.Fatalf("kernel %s: stopped at %q, not after \"=y\"", k, body[pos-2:min(len(body), pos+8)])
				}
			}
		})
	}
}
