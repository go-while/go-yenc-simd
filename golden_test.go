package yencsimd

import "testing"

// goldenCase is one call of rapidyenc's C decoder: input, start state, and
// what it returned (golden_data_test.go).
type goldenCase struct {
	in    string
	st    State
	out   string
	nSrc  int
	end   End
	stOut State
}

func TestGolden(t *testing.T) {
	for _, k := range kernels() {
		withKernel(k, func() {
			if got := Kernel(); got != k {
				t.Fatalf("kernel %s: Kernel() = %s", k, got)
			}
			for _, g := range golden {
				src := []byte(g.in)
				dst := make([]byte, len(src))
				st := g.st
				nd, ns, end, err := DecodeIncremental(dst, src, &st)
				if err != nil || string(dst[:nd]) != g.out || ns != g.nSrc || end != g.end || st != g.stOut {
					t.Fatalf("kernel %s, state %d, in %q:\n got  out=%q nSrc=%d end=%d state=%d err=%v\n want out=%q nSrc=%d end=%d state=%d",
						k, g.st, g.in, dst[:nd], ns, end, st, err, g.out, g.nSrc, g.end, g.stOut)
				}
			}
		})
	}
}

// The rules, readably. Every one is also in the golden table.
func TestRules(t *testing.T) {
	plain := func(s string) string { // what plain bytes decode to
		b := []byte(s)
		for i := range b {
			b[i] -= 42
		}
		return string(b)
	}
	for _, c := range []struct {
		name, in, out string
		end           End
		nSrc          int
	}{
		{"plain bytes lose 42", "abc", plain("abc"), EndNone, 3},
		{"line breaks are dropped", "ab\r\ncd", plain("abcd"), EndNone, 6},
		{"'=' escapes: the next byte loses 42+64", "a=}b", plain("a") + "\x13" + plain("b"), EndNone, 4},
		{"a stuffed dot at a line start is dropped", "a\r\n..b", plain("a.b"), EndNone, 6},
		{"a dot elsewhere is data", "a.b", plain("a.b"), EndNone, 3},
		{"\\r\\n=y ends the data, after the y", "ab\r\n=yend", plain("ab"), EndControl, 6},
		{"\\r\\n.=y (stuffed) ends it too", "ab\r\n.=yend", plain("ab"), EndControl, 7},
		{"\\r\\n.\\r\\n ends the article", "ab\r\n.\r\nxx", plain("ab"), EndArticle, 7},
		{"=y inside a line is an escaped y", "a=yb", plain("a") + "\x0f" + plain("b"), EndNone, 4},
	} {
		dst := make([]byte, len(c.in))
		st := StateCRLF
		nd, ns, end, _ := DecodeIncremental(dst, []byte(c.in), &st)
		if string(dst[:nd]) != c.out || end != c.end || ns != c.nSrc {
			t.Errorf("%s: %q -> %q end=%d nSrc=%d, want %q end=%d nSrc=%d", c.name, c.in, dst[:nd], end, ns, c.out, c.end, c.nSrc)
		}
	}
}

func TestDstTooSmall(t *testing.T) {
	st := StateCRLF
	if _, _, _, err := DecodeIncremental(make([]byte, 2), []byte("abc"), &st); err != ErrDstTooSmall {
		t.Fatalf("err = %v", err)
	}
}
