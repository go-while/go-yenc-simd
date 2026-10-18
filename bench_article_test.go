package yencsimd

import (
	"bufio"
	"bytes"
	"fmt"
	"hash/crc32"
	"io"
	"testing"
)

// benchArticle is benchBody as a whole article: the =ybegin line in front and
// a crc32= in the =yend line, so Decode and Reader check the CRC as they would
// for a real post. benchCRC is that CRC.
var benchArticle, benchCRC = func() ([]byte, uint32) {
	// decodeScalar: the vector kernels' tables are filled by init(), which
	// runs after the package-level variables
	dst := make([]byte, len(benchBody))
	st := StateCRLF
	nd, _, _ := decodeScalar(dst, benchBody, &st)
	crc := crc32.ChecksumIEEE(dst[:nd])
	body := benchBody[:bytes.LastIndex(benchBody, []byte("=yend"))]
	art := fmt.Sprintf("=ybegin line=128 size=%d name=bench.bin\r\n%s=yend size=%d crc32=%08x\r\n.\r\n", benchSize, body, benchSize, crc)
	return []byte(art), crc
}()

// BenchmarkDecodeArticle: Decode on the article in memory (decode, size and
// CRC check, one output buffer).
func BenchmarkDecodeArticle(b *testing.B) {
	for _, k := range kernels() {
		b.Run(k, func(b *testing.B) {
			withKernel(k, func() {
				b.SetBytes(int64(len(benchArticle)))
				b.ReportAllocs()
				for b.Loop() {
					data, m, err := Decode(benchArticle)
					if err != nil || len(data) != benchSize || m.CRC32 != benchCRC {
						b.Fatalf("err=%v, %d bytes, crc %08x", err, len(data), m.CRC32)
					}
				}
			})
		})
	}
}

// BenchmarkReader: NewReader on a bufio.Reader of 4 KiB (what net/textproto
// gives you) and of 64 KiB, read with a 32 KiB buffer, as io.Copy does.
func BenchmarkReader(b *testing.B) {
	for _, size := range []int{4 << 10, 64 << 10} {
		b.Run(fmt.Sprintf("%dKiB", size>>10), func(b *testing.B) {
			for _, k := range kernels() {
				b.Run(k, func(b *testing.B) {
					withKernel(k, func() {
						src := bytes.NewReader(benchArticle)
						br := bufio.NewReaderSize(src, size)
						buf := make([]byte, 32<<10)
						b.SetBytes(int64(len(benchArticle)))
						b.ReportAllocs()
						for b.Loop() {
							src.Reset(benchArticle)
							br.Reset(src)
							d := NewReader(br)
							n := 0
							for {
								got, err := d.Read(buf)
								n += got
								if err == io.EOF {
									break
								}
								if err != nil {
									b.Fatal(err)
								}
							}
							if n != benchSize || d.Meta().CRC32 != benchCRC {
								b.Fatalf("%d bytes, crc %08x", n, d.Meta().CRC32)
							}
						}
					})
				})
			}
		})
	}
}

// BenchmarkCRC32 is the floor under the article-level API: the CRC of the
// decoded data, which hash/crc32 computes.
func BenchmarkCRC32(b *testing.B) {
	data, _, err := Decode(benchArticle)
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(data)))
	for b.Loop() {
		if crc32.ChecksumIEEE(data) != benchCRC {
			b.Fatal("crc")
		}
	}
}
