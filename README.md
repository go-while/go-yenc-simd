# go-yenc-simd

A fast yEnc decoder for Go, for articles from Usenet (NNTP). Pure Go, no cgo.
With `GOEXPERIMENT=simd` on amd64 it uses AVX2 or AVX-512; everywhere else
it is plain Go.

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
| anything else | plain Go (about 0.4–1.1 GB/s) |

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

750 KB article, decode loop only:

| CPU | go-yenc-simd | rapidyenc (C) |
|---|---|---|
| Intel i7-10710U (AVX2), capped at 2 GHz | 1.8 GB/s | 3.7 GB/s |
| AMD EPYC 9B45 (Zen 5) | 11 GB/s (AVX-512), 7.4 GB/s (AVX2) | 17.4 GB/s |

## How it works, and tests

- A 7-state machine that matches rapidyenc's scalar reference byte for byte.
- Vector loops (16, 32 or 64 bytes) for the common bytes; `=` escapes, line
  breaks and stuffed dots are handled in the vector code too. Rare cases
  (`==`, escaped line breaks, the end markers) go through the state machine.
- Tests: 504 golden outputs from rapidyenc's C decoder on every vector path,
  each vector path against the state machine (random inputs and fuzzing),
  encode/decode round trips, and `Reader` on whole and byte-by-byte input.

```sh
GOEXPERIMENT=simd go test ./...
GOEXPERIMENT=simd go test -run '^$' -bench .
```

## Credits

The decoding rules follow [rapidyenc](https://github.com/animetosho/rapidyenc)
by animetosho (public domain / CC0).

## License

MIT, see [LICENSE](LICENSE).
