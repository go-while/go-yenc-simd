// Package yencsimd decodes yEnc articles from Usenet (NNTP), in pure Go,
// with SIMD on amd64 when built with GOEXPERIMENT=simd.
//
// Most programs need one of two functions:
//
//   - Decode(article) for an article in memory: it returns the data and its
//     Meta (name, size, part) and checks size and CRC32.
//   - NewReader(r) to stream an article from an NNTP connection: pass the
//     connection's *bufio.Reader after the "222" status line (with
//     net/textproto: tp.R, not tp.DotReader()).
//
// Both expect the article as the server sends it: CRLF lines, dot-stuffed,
// ending with a "." line.
//
// DecodeIncremental is the low-level step underneath, a drop-in for
// rapidyenc's rapidyenc_decode_incremental: it decodes the yEnc data in chunks
// of any size, undoes the dot-stuffing and stops at "=y" (the =yend line) or
// at the end of the article. Parsing the =y lines and the CRC are then up to
// the caller.
//
// The byte-wise decoder is a small state machine that matches rapidyenc's
// scalar reference (decoder.cc, do_decode_end_scalar<isRaw=true>) state for
// state. With GOEXPERIMENT=simd on amd64, vector loops decode 16, 32 or 64
// bytes per step (AVX, AVX2, AVX-512 VBMI2); elsewhere it is plain Go, 16
// bytes per step (two 64-bit words, SWAR).
package yencsimd

import (
	"encoding/binary"
	"errors"
	"math/bits"
	"unsafe"
)

// State is what the bytes before the current position were. The values are
// those of rapidyenc's RapidYencDecoderState, so they can be passed through.
type State int

const (
	StateCRLF     State = iota // "\r\n": start of a line (the initial state)
	StateEQ                    // "=": the next byte is escaped
	StateCR                    // "\r"
	StateNone                  // inside a line
	StateCRLFDT                // "\r\n."
	StateCRLFDTCR              // "\r\n.\r"
	StateCRLFEQ                // "\r\n=" or "\r\n.="
)

// End reports whether, and how, decoding stopped early.
type End int

const (
	EndNone    End = iota // all of src was decoded
	EndControl            // "\r\n=y" found: nSrc points after the 'y'
	EndArticle            // "\r\n.\r\n" found: nSrc points after the '\n'
)

// ErrDstTooSmall is returned when dst is shorter than src.
var ErrDstTooSmall = errors.New("yencsimd: destination must be at least the length of source")

// special marks the bytes that end a run of plain bytes.
var special = [256]bool{'=': true, '\r': true, '\n': true}

// DecodeIncremental decodes src into dst, starting in *state, and sets *state
// to the state after the last byte it read. It stops after an end sequence (see
// End). dst must be at least len(src): the output is never longer than the input.
// dst and src must not overlap: the decoder writes scratch bytes past the output
// position, so in-place decoding is not supported.
func DecodeIncremental(dst, src []byte, state *State) (nDst, nSrc int, end End, err error) {
	if len(dst) < len(src) {
		return 0, 0, EndNone, ErrDstTooSmall
	}
	nDst, nSrc, end = decode(dst, src, state)
	return nDst, nSrc, end, nil
}

// decodeScalar is the decoder without SIMD: the state machine, with a word
// loop (decodePlain) for everything inside a line. len(dst) >= len(src).
func decodeScalar(dst, src []byte, state *State) (int, int, End) {
	// decodePlain stores unchecked up to len(src): len(dst) >= len(src) is the
	// caller's contract (DecodeIncremental checks it); this panics if even the
	// capacity is short
	dst = dst[:len(src)]
	st := *state
	p, i, n := 0, 0, len(src)
	for i < n {
		if st == StateNone {
			p, i = decodePlain(dst, src, p, i)
			if i >= n {
				break
			}
		}
		var end End
		st, p, end = step(st, src[i], dst, p)
		i++
		if end != EndNone {
			*state = st
			return p, i, end
		}
	}
	*state = st
	return p, n, EndNone
}

// step runs byte c through the state machine in state st, writes its output
// (if any) at dst[p], and returns the new state and output position. It
// matches rapidyenc's scalar decoder (do_decode_end_scalar, raw) byte for byte.
func step(st State, c byte, dst []byte, p int) (State, int, End) {
	switch st {
	case StateCR:
		if c == '\n' {
			return StateCRLF, p, EndNone
		}
	case StateCRLF:
		switch c {
		case '.':
			return StateCRLFDT, p, EndNone
		case '=':
			return StateCRLFEQ, p, EndNone
		}
	case StateCRLFDT:
		switch c {
		case '\r':
			return StateCRLFDTCR, p, EndNone
		case '=':
			return StateCRLFEQ, p, EndNone
		}
		// "\r\n.X": the stuffed dot is dropped, X is an ordinary byte
	case StateCRLFDTCR:
		if c == '\n' {
			return StateCRLF, p, EndArticle
		}
	case StateCRLFEQ:
		if c == 'y' {
			return StateNone, p, EndControl
		}
		fallthrough // "\r\n=X" is an escaped X
	case StateEQ:
		dst[p] = c - 42 - 64
		if c == '\r' {
			return StateCR, p + 1, EndNone // an escaped \r still starts a line break
		}
		return StateNone, p + 1, EndNone
	}
	// an ordinary byte, in any state
	switch c {
	case '\r':
		return StateCR, p, EndNone
	case '\n':
		return StateNone, p, EndNone
	case '=':
		return StateEQ, p, EndNone
	}
	dst[p] = c - 42
	return StateNone, p + 1, EndNone
}

// Words of 8 equal bytes, for 8 bytes at a time (SWAR).
const (
	ones  = 0x0101010101010101
	highs = 0x8080808080808080
	eqs   = ones * '='
	crs   = ones * '\r'
	lfs   = ones * '\n'
	k42s  = ones * 42
)

// word decodes the 8 bytes of x: out has 42 subtracted from every byte, and
// the lowest set bit of t is bit 7 of the first byte that is '=', '\r' or
// '\n' (t is 0 if there is none; its higher bits mean nothing).
//
// A byte of x^eqs is 0 where x has '='. Minus 1, bit 7 of a byte is set
// where the byte was 0, where it was above 0x80 (x had a byte >= 0x80 other
// than '='|0x80) and in the bytes above the first 0, which get the borrow.
// hx has the high bits that were clear in x, so the AND with it keeps only
// the first case, and the bits below the first set one are exact. xh is x
// with the high bit of every byte set, so subtracting 42 from its bytes never
// borrows from the next byte: xh-k42s is x-42 in every byte, but with the
// high bit set, which hx puts right.
func word(x, ones, highs, eqs, crs, lfs, k42s uint64) (out, t uint64) {
	xh := x | highs
	hx := xh ^ x
	t = (x ^ eqs - ones) | (x ^ crs - ones) | (x ^ lfs - ones)
	return (xh - k42s) ^ hx, t & hx
}

// decodePlain is decodeScalar in StateNone, from src[i] to dst[p]: plain bytes
// 16 at a time, and escapes and line breaks that leave the state StateNone,
// without the state machine. It returns the positions where the state machine
// has to take over, or len(src). len(dst) >= len(src).
//
// What it does without the state machine is what the state machine does:
// "=X" is an escaped X, unless X is '\r' (an escaped \r still starts a line
// break); after "\r\n" a byte other than '.', '=', '\r' and '\n' is an
// ordinary byte, so the line break is just dropped.
//
// The words with the bytes minus 42 are stored before the first special byte
// is known, so dst gets up to 8 bytes past the plain run that are
// overwritten later. The stores are unchecked (a checked store needs len(dst)
// in a register, and with it the word loop spilled): the output never gets
// ahead of the input, p <= i, so p+16 <= i+16 <= n <= len(dst) in the word
// loop, p+8 <= i+8 <= n in the 8-byte step and p <= i < n for the bytes.
func decodePlain(dst, src []byte, p, i int) (int, int) {
	n := len(src)
	ones, highs := swarWords() // the constants as values, see swarWords
	eqs, crs, lfs, k42s := ones*'=', ones*'\r', ones*'\n', ones*42
	dp := unsafe.Pointer(unsafe.SliceData(dst)) // unchecked: see above
	for i < n-15 {
		// the first word, and the second only if the first has no special
		// byte, so a short run costs one word
		out0, t0 := word(binary.LittleEndian.Uint64(src[i:]), ones, highs, eqs, crs, lfs, k42s)
		binary.LittleEndian.PutUint64((*[8]byte)(unsafe.Add(dp, p))[:], out0)
		// the first special byte, if any: an escape or a line break (or
		// something rarer) at src[i+k]
		var k int
		if t0 != 0 {
			k = bits.TrailingZeros64(t0) >> 3
		} else {
			out1, t1 := word(binary.LittleEndian.Uint64(src[i+8:]), ones, highs, eqs, crs, lfs, k42s)
			binary.LittleEndian.PutUint64((*[8]byte)(unsafe.Add(dp, p+8))[:], out1)
			if t1 == 0 {
				i += 16
				p += 16
				continue
			}
			k = 8 + bits.TrailingZeros64(t1)>>3
		}
		i += k
		p += k
		if i+2 >= n {
			break
		}
		s := src[i : i+3] // one bounds check
		c, d, f := s[0], s[1], s[2]
		if c == '=' && d != '\r' {
			*(*byte)(unsafe.Add(dp, p)) = d - 42 - 64
			p++
			i += 2
		} else if c == '\r' && d == '\n' && f != '.' && f != '=' && f != '\r' && f != '\n' {
			i += 2
		} else {
			return p, i
		}
	}
	// 8 more bytes, up to the first special byte
	if i < n-7 {
		out, t := word(binary.LittleEndian.Uint64(src[i:]), ones, highs, eqs, crs, lfs, k42s)
		binary.LittleEndian.PutUint64((*[8]byte)(unsafe.Add(dp, p))[:], out)
		k := bits.TrailingZeros64(t) >> 3 // 8 if none
		i += k
		p += k
	}
	// the last bytes, one at a time
	for i < n {
		c := src[i]
		if special[c] {
			break
		}
		*(*byte)(unsafe.Add(dp, p)) = c - 42
		i++
		p++
	}
	return p, i
}
