package yencsimd_test

import (
	"bufio"
	"fmt"
	"io"
	"net/textproto"
	"strings"

	yencsimd "github.com/go-while/go-yenc-simd"
)

// The article as a server sends it after "222 ... body follows":
// "Hello, yEnc!" encoded, dot-stuffed, ending in a "." line.
const article = "=ybegin line=128 size=12 name=hello.txt\r\n" +
	"r\x8f\x96\x96\x99VJ\xa3o\x98\x8dK\r\n" +
	"=yend size=12 crc32=0337ab3d\r\n" +
	".\r\n"

// Decode a whole article you already have in memory.
func ExampleDecode() {
	data, meta, err := yencsimd.Decode([]byte(article))
	if err != nil {
		fmt.Println("error:", err) // ErrCRC, ErrSize, ErrTruncated or ErrNoYbegin
		return
	}
	fmt.Printf("%s: %q (%d bytes)\n", meta.Name, data, meta.Size)
	// Output: hello.txt: "Hello, yEnc!" (12 bytes)
}

// Stream an article straight from an NNTP connection (net/textproto).
// Pass the connection's bufio.Reader (tp.R), not tp.DotReader(): Reader
// handles the dot-stuffing and the closing "." line itself, and leaves the
// connection at the next response.
func ExampleNewReader() {
	// a stand-in for the server's reply to "BODY <id>"
	tp := textproto.NewReader(bufio.NewReader(strings.NewReader("222 0 <id@example.test> body\r\n" + article)))
	if _, _, err := tp.ReadCodeLine(222); err != nil {
		fmt.Println("error:", err)
		return
	}
	d := yencsimd.NewReader(tp.R)
	var out strings.Builder
	if _, err := io.Copy(&out, d); err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("%s: %q\n", d.Meta().Name, out.String())
	// Output: hello.txt: "Hello, yEnc!"
}

// The low-level step: decode yEnc data (the lines between =ybegin and =yend)
// in chunks of any size. Reader is built on this.
func ExampleDecodeIncremental() {
	body := []byte("r\x8f\x96\x96\x99VJ\xa3o\x98\x8dK\r\n=yend size=12\r\n")
	st := yencsimd.StateCRLF // the data starts at a line start
	dst := make([]byte, len(body))
	n, used, end, _ := yencsimd.DecodeIncremental(dst, body, &st)
	fmt.Printf("%q, stopped at %q (end=%v)\n", dst[:n], body[used-2:used+3], end == yencsimd.EndControl)
	// Output: "Hello, yEnc!", stopped at "=yend" (end=true)
}
