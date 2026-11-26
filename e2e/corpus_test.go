package e2e

import (
	"bytes"
	"errors"
	"testing"

	yencsimd "github.com/go-while/go-yenc-simd"
	"github.com/go-while/nntp-testkit/fakeserver"
	"github.com/go-while/nntp-testkit/yenc"
)

// TestCorpus serves every case of nntp-testkit's yenc.Corpus (broken and odd posts, each with
// the outcome a decoder should give) as a post, fetches it with BODY and decodes it with
// Reader, for every kernel. After each case a good post is fetched on the same connection, so
// the connection must still be in step.
func TestCorpus(t *testing.T) {
	cases := yenc.Corpus()
	if len(cases) < 5 {
		t.Fatalf("yenc.Corpus has %d cases, want at least the 5 of v0.15.0", len(cases))
	}
	st := fakeserver.NewStore()
	st.AddGroup("yenc.corpus", "odd and broken posts")
	good := yenc.Data(yenc.Random, 1000, 9)
	goodN := st.AddArticle("yenc.corpus", article("good", yenc.Encode("good.bin", good, yenc.Options{})))
	nums := make([]int64, len(cases))
	for i, k := range cases {
		nums[i] = st.AddArticle("yenc.corpus", article("case-"+k.Name, k.Lines))
	}
	s := fakeserver.New(t, fakeserver.Options{Store: st})
	eachKernel(t, func(t *testing.T) {
		c := mustDial(t, s.Addr(), 64<<10)
		c.group(t, "yenc.corpus")
		for i, k := range cases {
			data, _, err := c.body(nums[i])
			t.Logf("%s (want %s): %d bytes, err %v", k.Name, k.Want, len(data), err)
			switch k.Want {
			case fakeserver.WantAccept:
				if err != nil || !bytes.Equal(data, k.Data) {
					t.Errorf("%s: want the data accepted; got %d bytes, err %v (%s)", k.Name, len(data), err, k.Why)
				}
			case fakeserver.WantError:
				// The decoder's own error, not a failed BODY command or a broken server.
				if !errors.Is(err, yencsimd.ErrCRC) && !errors.Is(err, yencsimd.ErrSize) &&
					!errors.Is(err, yencsimd.ErrTruncated) && !errors.Is(err, yencsimd.ErrNoYbegin) {
					t.Errorf("%s: want a decoder error; got %d bytes, err %v (%s)", k.Name, len(data), err, k.Why)
				}
			default: // WantEither
				if err == nil && !bytes.Equal(data, k.Data) {
					t.Errorf("%s: accepted, but the data differ (%d bytes, want %d)", k.Name, len(data), len(k.Data))
				}
			}
			got, _, err := c.body(goodN)
			if err != nil || !bytes.Equal(got, good) {
				t.Fatalf("after %s: the next fetch on the connection failed: %d bytes, %v", k.Name, len(got), err)
			}
		}
	})
}
