# go-yenc-simd

A fast yEnc decoder for Go, for articles from Usenet (NNTP). Pure Go, no cgo.
With `GOEXPERIMENT=simd` it uses AVX, AVX2 or AVX-512 on amd64 and NEON on
arm64; everywhere else it is plain Go.

```sh
go get github.com/go-while/go-yenc-simd
```

## Decode an article

You have the article body as the server sent it (lines ending in `\r\n`,
ending with a line holding a single `.`):

```go
import yencsimd "github.com/go-while/go-yenc-simd"

data, meta, err := yencsimd.Decode(article)
if err != nil {
	// yencsimd.ErrCRC, ErrSize, ErrTruncated or ErrNoYbegin
}
fmt.Println(meta.Name, len(data))
```

`Decode` finds the `=ybegin` line (headers before it are skipped, so an
`ARTICLE` response works too), decodes the data, and checks the size and the
CRC32 against the `=yend` line.

## Stream from an NNTP connection

Send `BODY <message-id>`, read the `222` status line, then hand the
connection's buffered reader to `NewReader`:

```go
tp := textproto.NewConn(conn)
id, _ := tp.Cmd("BODY %s", msgid)
tp.StartResponse(id)
if _, _, err := tp.ReadCodeLine(222); err != nil {
	return err
}
d := yencsimd.NewReader(tp.R)   // tp.R, NOT tp.DotReader()
_, err := io.Copy(file, d)      // io.EOF -> nil: size and CRC were checked
tp.EndResponse(id)
meta := d.Meta()                // Name, Size, Part, Total, Begin, End, CRC32
```

`NewReader` reads through the closing `.` line, so the connection is ready
for the next command.

Each step decodes what the `bufio.Reader` has buffered (never reading past
the article), straight into the caller's buffer when that can take it or
holds 8 KiB or more (`io.Copy`'s 32 KiB buffer does). So the buffer size sets how much
work a step does and how often the connection is read: `textproto.NewConn`
uses the 4 KiB default; `textproto.NewReader(bufio.NewReaderSize(conn,
64<<10))` makes bigger steps and up to 16 times fewer reads.

**Don't use `tp.DotReader()`.** It removes NNTP's dot-stuffing (`..` at a
line start → `.`), and the decoder does that too, so a line that really
starts with a dot would lose it.

## Multi-part files

Each article is one part. `Meta` says where it goes:

```go
data, meta, err := yencsimd.Decode(article)
// meta.Part of meta.Total, bytes meta.Begin..meta.End (1-based, inclusive)
// of a file of meta.Size bytes
f.WriteAt(data, meta.Begin-1)
```

For a part, the CRC checked is the `pcrc32=` of that part.

## Build with SIMD

```sh
GOEXPERIMENT=simd go build ./...
```

| | |
|---|---|
| amd64 with `GOEXPERIMENT=simd` | AVX-512 VBMI2, AVX2 or AVX, chosen at run time |
| arm64 with `GOEXPERIMENT=simd` | NEON, 64 bytes per step |
| anything else | plain Go, 16 bytes per step (about 1–3.5 GB/s) |

`GOEXPERIMENT=simd` needs Go 1.27 and is an experiment, outside the Go 1
compatibility promise. Without it the package builds and works the same,
only slower. `yencsimd.Kernel()` tells you which path runs.

## Low level: DecodeIncremental

`Reader` is built on `DecodeIncremental`, a drop-in for rapidyenc's
`rapidyenc_decode_incremental`. Use it if you parse the `=y` lines yourself
or feed data in your own chunks:

```go
st := yencsimd.StateCRLF                 // the data starts at a line start
dst := make([]byte, len(chunk))          // output is never longer than input
n, used, end, err := yencsimd.DecodeIncremental(dst, chunk, &st)
// dst[:n]: decoded data; chunk[:used] was consumed
// end == EndControl: "=y" reached; chunk[used-2:] starts the =yend line
// end == EndArticle: "\r\n.\r\n" reached (the article ended without =yend)
// end == EndNone:    all of chunk decoded; pass the rest with the same st
```

It expects the data between the `=ybegin`/`=ypart` lines and `=yend`, still
dot-stuffed. It does not check the CRC.

## Speed

750 KB article, median of 3 pinned runs of `go test -bench .`, in MB/s of
input. rapidyenc is the C reference (animetosho/rapidyenc, gcc 14, same input,
same core).

| | Intel i7-10710U (AVX2), fixed 2 GHz | AMD EPYC 9B45 (Zen 5, AVX-512) |
|---|---|---|
| `DecodeIncremental`, random data | 2.9 GB/s | 24.7 GB/s (AVX-512), 15.1 GB/s (AVX2) |
| same, data that needs no escapes | 4.8 GB/s | 26.0 GB/s, 22.7 GB/s |
| same, an escape in every second byte | 3.0 GB/s | 21.9 GB/s, 11.3 GB/s |
| same, plain Go (no `GOEXPERIMENT=simd`) | 0.95 GB/s | 3.4 GB/s |
| `Decode` (whole article, size and CRC checked) | 1.8 GB/s | 10.4 GB/s |
| `Reader` on a 64 KiB `bufio.Reader` | 2.0 GB/s | 12.6 GB/s |
| rapidyenc, random data | 3.8 GB/s | 17.3 GB/s (VBMI2), 18.8 GB/s (AVX2) |

`Decode` and `Reader` include the CRC32 (`hash/crc32`: 10 GB/s on the i7,
33 GB/s on the EPYC), which is now a large share of their time.

On arm64 (NEON) the numbers are from GitHub's runners: `yencbench -time
500ms` in five CI runs. They are shared VMs, not pinned, so the Apple M1
numbers vary a lot between runs:

| | Apple M1 (`macos-latest`) | Neoverse N2 (`ubuntu-24.04-arm`) |
|---|---|---|
| `DecodeIncremental`, random data | 6.0–8.0 GB/s | 5.4 GB/s |
| same, data that needs no escapes | 7.0–11.8 GB/s | 7.7 GB/s |
| same, an escape in every second byte | 6.2–9.2 GB/s | 5.8 GB/s |
| same, plain Go (no `GOEXPERIMENT=simd`) | 1.9–2.7 GB/s | 2.6–2.7 GB/s |
| `Decode` (whole article, size and CRC checked) | 2.1–3.2 GB/s | 3.4 GB/s |

One run of `macos-latest` got an Apple M2 Pro instead: 8.9 GB/s on random
data, plain Go 2.8 GB/s.

## Benchmarks

The quickest way, no clone needed (Go 1.27):

```sh
GOEXPERIMENT=simd go run github.com/go-while/go-yenc-simd/cmd/yencbench@latest
```

It prints MB/s for every kernel your CPU can run (AVX-512 VBMI2, AVX2, AVX or
NEON, and scalar) on three 750 KB articles (random data, data that needs no
escapes, an escape in every second byte), and for `Decode` of a whole article
with the CRC check. Without `GOEXPERIMENT=simd` it measures the plain Go
build. Flags: `-time 1s` per measurement, `-size 750000` data bytes per
article, `-kernels avx2,scalar` to pick kernels. Pin it to one core for steady
numbers (`taskset -c 2 ...` on Linux).

From a program, for example behind a `-yencbench` flag:

```go
import "github.com/go-while/go-yenc-simd/yencbench"

if *bench {
	if err := yencbench.Run(os.Stdout, yencbench.Options{}); err != nil {
		log.Fatal(err)
	}
	return
}
```

`yencbench.Measure` returns the numbers instead of printing them; the kernel
in use is restored afterwards. `yencsimd.Kernels()` lists the kernels and
`yencsimd.SetKernel` picks one (only while nothing is decoding).

The full set, including `Reader` with 4 KiB and 64 KiB buffers, is in the
package's tests:

```sh
git clone https://github.com/go-while/go-yenc-simd && cd go-yenc-simd
GOEXPERIMENT=simd go test -run '^$' -bench . -benchtime 1s
```

## How it works, and tests

- A 7-state machine that matches rapidyenc's scalar reference byte for byte.
- Vector loops (16, 32 or 64 bytes) for the common bytes; `=` escapes, line
  breaks and stuffed dots are handled in the vector code too. Rare cases
  (`==`, escaped line breaks, the end markers) go through the state machine.
- Tests: 504 golden outputs from rapidyenc's C decoder on every vector path,
  each vector path against the state machine (random inputs; every string of
  up to 4 of `=`, `\r`, `\n`, `.`, `y` and a plain byte, at every offset from
  0 to 140, so across every vector boundary; fuzzing), encode/decode round
  trips, and `Reader` on whole and byte-by-byte input. On amd64 a test reads
  the machine code and checks that each kernel uses only instructions its CPU
  check guarantees (an AVX2 instruction in the AVX kernel would crash Sandy
  Bridge).

```sh
GOEXPERIMENT=simd go test ./...
```

## Credits

The decoding rules follow [rapidyenc](https://github.com/animetosho/rapidyenc)
by animetosho (public domain / CC0).

## License

MIT, see [LICENSE](LICENSE).
