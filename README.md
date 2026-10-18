# go-yenc-simd

A pure-Go yEnc decoder for raw NNTP article bodies, with SIMD on amd64
through Go's `simd/archsimd` package. No cgo.

It is a drop-in for [rapidyenc](https://github.com/animetosho/rapidyenc)'s
`rapidyenc_decode_incremental`: same input, output, states and end codes.
It undoes NNTP dot-stuffing (`\r\n..` → `\r\n.`) and stops at the end of the
yEnc data (`\r\n=y`, also dot-stuffed as `\r\n.=y`) or of the article
(`\r\n.\r\n`). Parsing `=ybegin`/`=ypart`/`=yend` and checking the CRC is up to
the caller.

```go
import yencsimd "github.com/go-while/go-yenc-simd"

st := yencsimd.StateCRLF           // the body starts at a line start
dst := make([]byte, len(chunk))    // output is never longer than the input
n, used, end, err := yencsimd.DecodeIncremental(dst, chunk, &st)
// dst[:n] is decoded data; chunk[:used] was consumed.
// end == EndControl: "=y" reached, chunk[used-2:] starts the =yend line.
// end == EndArticle: "\r\n.\r\n" reached. EndNone: feed the next chunk with st.
```

## Build

| | |
|---|---|
| `GOEXPERIMENT=simd go build` on amd64 | AVX-512 VBMI2 (64 bytes/step, `VPCOMPRESSB`), AVX2 (32), AVX (16), picked at run time |
| any other build or CPU | plain Go |

`GOEXPERIMENT=simd` needs Go 1.27 and is outside the Go 1 compatibility
promise. `Kernel()` reports the path in use.

## How it works

- `step` is a 7-state machine that matches rapidyenc's scalar reference
  (`decoder.cc`, `do_decode_end_scalar<isRaw=true>`) byte for byte.
- The vector loops run in the "inside a line" state. Each block gets its
  escapes from a load one byte back and its line starts from two bytes back,
  so blocks may start and end anywhere. Bytes to drop (`=`, `\r`, `\n`, a
  stuffed dot) are packed out with `VPSHUFB` and 256-entry tables, or with
  `VPCOMPRESSB` on AVX-512 VBMI2.
- Anything else goes through the state machine one byte at a time: `==`, an
  escaped line break (invalid yEnc; rapidyenc has quirks for it, and they are
  kept), and the end sequences.

## Tests

- `golden_data_test.go`: 504 calls of rapidyenc's C decoder (72 tricky
  inputs × 7 start states), checked on every path the CPU has.
- Every vector path against the state machine: 50,000 random inputs dense in
  `= \r \n . y` per path, and a fuzz target.
- Encode → dot-stuff → decode round trips in random chunks, including
  escape-heavy data.

```sh
GOEXPERIMENT=simd go test ./...
GOEXPERIMENT=simd go test -run '^$' -fuzz FuzzVectorPathsMatchStateMachine -fuzztime 60s
GOEXPERIMENT=simd go test -run '^$' -bench .
```

## Speed

750 KB article body, decode loop only:

| CPU | go-yenc-simd | rapidyenc (C) |
|---|---|---|
| Intel i7-10710U (AVX2), capped at 2 GHz | 1.8 GB/s (AVX2) | 3.7 GB/s |
| AMD EPYC 9B45 (Zen 5, KVM) | 11 GB/s (AVX-512 VBMI2), 7.4 GB/s (AVX2) | 17.4 GB/s |
| plain Go, same CPUs | 0.4 / 1.1 GB/s | |

Inside a full decoder (header parsing, CRC32, 4–64 KiB reads) the gap to C
was 61–69% (i7) and 78–81% (EPYC).

## Credits

The decoding rules follow rapidyenc by animetosho (public domain / CC0).

## License

MIT, see [LICENSE](LICENSE).
