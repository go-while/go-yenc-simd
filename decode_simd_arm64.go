//go:build goexperiment.simd && arm64

package yencsimd

import (
	"errors"
	"simd/archsimd"
	"sync"
	"unsafe"
)

// hasNEON is whether decode uses the NEON loops. Every arm64 CPU has NEON
// (Advanced SIMD is part of ARMv8-A), so it starts true; SetKernel("scalar")
// turns it off.
var hasNEON = true

// Kernel reports which fast path the decoder uses.
func Kernel() string {
	if hasNEON {
		return "neon"
	}
	return "scalar"
}

// Kernels lists the decode paths this CPU can run, fastest first.
func Kernels() []string { return []string{"neon", "scalar"} }

// SetKernel limits the decoder to the named path, one of Kernels(). It is
// meant for benchmarks and tests: it changes package state, so call it only
// while nothing is decoding.
func SetKernel(name string) error {
	switch name {
	case "neon":
		hasNEON = true
		return nil
	case "scalar":
		hasNEON = false
		return nil
	}
	return errors.New("yencsimd: kernel " + name + " is not available on this CPU")
}

func decode(dst, src []byte, state *State) (int, int, End) {
	if hasNEON {
		return decodeNEON(dst, src, state)
	}
	return decodeScalar(dst, src, state)
}

// decodeNEON is the state machine with NEON loops, as decodeSIMD is on amd64
// (the rules are the same, see there): once in StateNone it decodes 64 bytes
// per step (steps64), and 16 (steps16) near the end of src and for a step
// that steps64 leaves to it. A vector with a bad byte ends the loops; the
// state machine takes it from there, one byte at a time. The last bytes, too
// few for a vector, go through decodePlain, the word loop of the scalar path.
//
// The design follows rapidyenc's NEON decoder (animetosho/rapidyenc) and its
// archsimd port in mnightingale/rapidyenc, both on GitHub: 64 bytes as four
// vectors, '\r' and '\n' found with one table lookup, and a 2^15-entry table
// that packs the bytes to keep.
func decodeNEON(dst, src []byte, state *State) (int, int, End) {
	neonOnce.Do(fillNEON)
	st := *state
	p, i, n := 0, 0, len(src)
	for i < n {
		if st == StateNone && i+17 > n {
			// the tail, too short for a vector: the word loop of the scalar
			// path, up to a byte that needs the state machine
			p, i = decodePlain(dst, src, p, i)
			if i == n {
				break
			}
		} else if st == StateNone && i >= 2 && src[i-1] != '=' {
			// the vector loops need the bytes before i to mean what they say
			// (see decodeSIMD)
			start := i
			for {
				p, i = steps64(dst, src, p, i)
				// the step it stopped at, or the tail of src: steps16 decodes
				// it, up to a vector with a bad byte
				end := i + 64
				p, i = steps16(dst, src, p, i, end)
				if i < end {
					break
				}
			}
			if i != start {
				st = stateAfter(src, i)
				continue
			}
		}
		// one byte through the state machine
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

// NEON has no instruction that gathers the top bits of a compare result
// into an integer (x86's PMOVMSKB). The loops make one: each 0xff byte is
// ANDed with its weight in neonWeights, and pairwise adds of 16-bit lanes
// (ADDP) sum the 8 lanes of a vector into one. The low byte of a lane holds
// the even bytes and the high byte the odd ones, so bit j of the 16-bit mask
// is byte 2j and bit 8+j is byte 2j+1: the bytes in the order 0, 2, 4, ...
// 14, 1, 3, ... 15. neonPack is indexed in that order; byte 15 is bit 15,
// as on amd64.
var neonWeights = [16]uint8{1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128}

// neonCRLF is a TBL table that gives 0xff for '\r' and '\n': TBL gives 0
// for an index of 16 or more, so every other byte gives 0.
var neonCRLF = [16]uint8{'\n': 0xff, '\r': 0xff}

// neonPack holds the compaction tables, filled by fillNEON before the first
// decode. idx[m] are TBL indices that pack the bytes not dropped by the low
// 15 bits of mask m (in the order above) to the front, byte 15 right after
// them, as pack.idx does on amd64; cnt[m] is the number of bytes m keeps.
var neonPack struct {
	idx [1 << 15][16]uint8
	cnt [1 << 16]uint8
}

var neonOnce sync.Once

// neonIdx and neonCnt point to the tables (see steps64).
var neonIdx, neonCnt = &neonPack.idx, &neonPack.cnt

// fillNEON fills neonPack as fillPack fills pack on amd64 (idx[m] is byte 0
// if m keeps it, then idx[m>>1] with every index one higher), with m in the
// order of the bytes, and stores each entry at neonOrder(m): 0.4 ms on a
// 2 GHz Comet Lake, against 3.4 ms for a loop over the bytes of each entry.
func fillNEON() {
	const ones = 0x0101010101010101
	for k := range 16 {
		neonPack.idx[0][k] = uint8(k)
	}
	for m := 1; m < 1<<15; m++ {
		src := (*[2]uint64)(unsafe.Pointer(&neonPack.idx[neonOrder(m>>1)]))
		lo, hi := src[0]+ones, src[1]+ones
		if m&1 == 0 {
			lo, hi = lo<<8, hi<<8|lo>>56
		}
		dst := (*[2]uint64)(unsafe.Pointer(&neonPack.idx[neonOrder(m)]))
		dst[0], dst[1] = lo, hi
	}
	neonPack.cnt[0] = 16
	for m := 1; m < 1<<16; m++ {
		neonPack.cnt[m] = neonPack.cnt[m>>1] - uint8(m&1)
	}
}

// neonOrder is mask m (bit b for byte b) in the order of neonWeights: the
// even bits in the low byte, the odd bits in the high byte. Three delta
// swaps move bits 1, 3, 5 and 7 of each byte past the even ones.
func neonOrder(m int) int {
	x := uint32(m)
	t := (x ^ x>>1) & 0x2222
	x ^= t ^ t<<1
	t = (x ^ x>>2) & 0x0c0c
	x ^= t ^ t<<2
	t = (x ^ x>>4) & 0x00f0
	x ^= t ^ t<<4
	return int(x)
}

// v8 is a compare result as bytes: 0xff where true, 0 where false.
func v8(m archsimd.Mask8x16) archsimd.Uint8x16 { return m.ToInt8x16().ToBits() }

// nonzero reports whether x has a byte that is not 0: UQXTN narrows the two
// 64-bit halves to 32 bits (saturating, so what is not 0 stays so), and one
// move to an integer register tests the 64 bits.
func nonzero(x archsimd.Uint8x16) bool {
	return x.ReshapeToUint64s().SaturateToUint32().ReshapeToUint64s().GetElem(0) != 0
}

// steps64 decodes 64 bytes per step from src[i] while 65 bytes are left. It
// stops at a step with anything but plain bytes, escapes ("=" + byte) and
// line breaks that need no more than dropping: steps16 decodes that one.
// The caller guarantees the state machine invariant at i (see decodeSIMD);
// the bytes before i may be a '=' or "\r\n" (where steps16 stopped).
//
// A step is four 16-byte vectors. If none has a '=', '\r' or '\n' and the
// byte before the step is plain (notPlain), the output is the 64 bytes minus
// 42. Otherwise '=', '\r' and '\n' are dropped, the bytes after a '=' lose
// 42+64 and the others 42, and a TBL from neonPack packs each vector. That
// is all unless a '=' or '\n' comes before a '=', '.', '\r' or '\n': a line
// can start with a stuffed dot or an end sequence, "==", an escaped \r or \n
// are bad bytes (see decodeSIMD), and "=.", "\n\r", "\n\n" do not occur in
// yEnc data. steps64 leaves such a step to steps16, which has all the rules.
//
// Bounds (unchecked; i+65 <= len(src) <= len(dst), p <= i, i >= 2): the
// loads are src[i-1:i+64), the stores reach dst[p+64).
func steps64(dst, src []byte, p, i int) (int, int) {
	if len(src) < 67 {
		return p, i
	}
	eq := archsimd.BroadcastUint8x16('=')
	lf := archsimd.BroadcastUint8x16('\n')
	dot := archsimd.BroadcastUint8x16('.')
	k42 := archsimd.BroadcastUint8x16(42)
	k64 := archsimd.BroadcastUint8x16(64)
	crlf := archsimd.LoadUint8x16Array(&neonCRLF)
	wt := archsimd.LoadUint8x16Array(&neonWeights)
	// the tables through pointers loaded once, so that the loop keeps them in
	// registers instead of rebuilding their addresses (two instructions each)
	idxt, cntt := neonIdx, neonCnt
	// the nil checks, here and not in the loop
	_, _ = idxt[0], cntt[0]
	sq := unsafe.Add(unsafe.Pointer(unsafe.SliceData(src)), i) // &src[i]
	dq := unsafe.Add(unsafe.Pointer(unsafe.SliceData(dst)), p) // &dst[p]
	last := unsafe.Add(unsafe.Pointer(unsafe.SliceData(src)), len(src)-65)
	// the byte before the step: an escape or a line start needs the full step
	notPlain := uint64(0)
	if c := src[i-1]; c == '=' || c == '\n' {
		notPlain = 1
	}
	// The fast path is tried in a chunk of steps only if in the chunk before
	// at least 5 steps in 16 had nothing to drop: with random data it is one
	// step in five, and the branch on it, mispredicted that often, cost more
	// than it saved (2-5% on a Neoverse N2, 4-20% on the Apple M1 of GitHub's
	// runners). The first chunk, 8 steps, does without; the others are 32.
	plain, steps := 0, 1 // steps with nothing to drop, of the steps in the last chunk
	size := 8
loop:
	for uintptr(sq) <= uintptr(last) {
		tryFast := plain*16 >= steps*5
		plain = 0
		begin := sq
		end := last
		if uintptr(last)-uintptr(sq) > uintptr(size-1)*64 {
			end = unsafe.Add(sq, (size-1)*64)
		}
		for uintptr(sq) <= uintptr(end) {
			v0 := archsimd.LoadUint8x16Array(at16(sq, 0))
			v1 := archsimd.LoadUint8x16Array(at16(sq, 16))
			v2 := archsimd.LoadUint8x16Array(at16(sq, 32))
			v3 := archsimd.LoadUint8x16Array(at16(sq, 48))
			// '=', '\r', '\n': never output
			e0, e1, e2, e3 := v8(v0.Equal(eq)), v8(v1.Equal(eq)), v8(v2.Equal(eq)), v8(v3.Equal(eq))
			c0 := crlf.LookupOrZero(v0).Or(e0)
			c1 := crlf.LookupOrZero(v1).Or(e1)
			c2 := crlf.LookupOrZero(v2).Or(e2)
			c3 := crlf.LookupOrZero(v3).Or(e3)
			if tryFast && !nonzero(c0.Or(c1).Or(c2).Or(c3)) && notPlain == 0 {
				v0.Sub(k42).StoreArray((*[16]uint8)(dq))
				v1.Sub(k42).StoreArray(at16(dq, 16))
				v2.Sub(k42).StoreArray(at16(dq, 32))
				v3.Sub(k42).StoreArray(at16(dq, 48))
				sq = unsafe.Add(sq, 64)
				dq = unsafe.Add(dq, 64)
				plain++
				continue
			}
			// the byte before each byte
			b0 := archsimd.LoadUint8x16Array(at16(sq, -1))
			b1 := archsimd.LoadUint8x16Array(at16(sq, 15))
			b2 := archsimd.LoadUint8x16Array(at16(sq, 31))
			b3 := archsimd.LoadUint8x16Array(at16(sq, 47))
			s0, s1, s2, s3 := v8(b0.Equal(eq)), v8(b1.Equal(eq)), v8(b2.Equal(eq)), v8(b3.Equal(eq)) // escaped
			// a '=' or '\n' before a '=', '.', '\r' or '\n'
			r0 := s0.Or(v8(b0.Equal(lf))).And(c0.Or(v8(v0.Equal(dot))))
			r1 := s1.Or(v8(b1.Equal(lf))).And(c1.Or(v8(v1.Equal(dot))))
			r2 := s2.Or(v8(b2.Equal(lf))).And(c2.Or(v8(v2.Equal(dot))))
			r3 := s3.Or(v8(b3.Equal(lf))).And(c3.Or(v8(v3.Equal(dot))))
			if nonzero(r0.Or(r1).Or(r2).Or(r3)) {
				break loop
			}
			// escaped bytes lose 42+64, the others 42: s*64 is -64 where s is 0xff
			o0 := s0.MulAdd(k64, v0.Sub(k42))
			o1 := s1.MulAdd(k64, v1.Sub(k42))
			o2 := s2.MulAdd(k64, v2.Sub(k42))
			o3 := s3.MulAdd(k64, v3.Sub(k42))
			// the four 16-bit drop masks in one 64-bit word, see neonWeights
			w01 := c0.And(wt).ReshapeToUint16s().ConcatAddPairs(c1.And(wt).ReshapeToUint16s())
			w23 := c2.And(wt).ReshapeToUint16s().ConcatAddPairs(c3.And(wt).ReshapeToUint16s())
			w := w01.ConcatAddPairs(w23)
			m := w.ConcatAddPairs(w).ReshapeToUint64s().GetElem(0)
			if m == 0 {
				plain++
			}
			o0.LookupOrZero(archsimd.LoadUint8x16Array(&idxt[m&0x7fff])).StoreArray((*[16]uint8)(dq))
			dq = unsafe.Add(dq, cntt[m&0xffff])
			o1.LookupOrZero(archsimd.LoadUint8x16Array(&idxt[m>>16&0x7fff])).StoreArray((*[16]uint8)(dq))
			dq = unsafe.Add(dq, cntt[m>>16&0xffff])
			o2.LookupOrZero(archsimd.LoadUint8x16Array(&idxt[m>>32&0x7fff])).StoreArray((*[16]uint8)(dq))
			dq = unsafe.Add(dq, cntt[m>>32&0xffff])
			o3.LookupOrZero(archsimd.LoadUint8x16Array(&idxt[m>>48&0x7fff])).StoreArray((*[16]uint8)(dq))
			dq = unsafe.Add(dq, cntt[m>>48])
			notPlain = m >> 63 // the last byte was dropped: a '=' or line break
			sq = unsafe.Add(sq, 64)
		}
		steps = int(uintptr(sq)-uintptr(begin)) >> 6
		size = 32
	}
	return int(uintptr(dq) - uintptr(unsafe.Pointer(unsafe.SliceData(dst)))),
		int(uintptr(sq) - uintptr(unsafe.Pointer(unsafe.SliceData(src))))
}

// steps16 decodes one 16-byte vector per step from src[i] while 17 bytes
// are left, up to end, and stops at a vector with a bad byte. It has all the
// rules of steps16 on amd64; the bytes before i may be a '=' or "\r\n"
// (where steps64 stopped).
//
// Bounds (unchecked; i+17 <= len(src) <= len(dst), p <= i, i >= 2): a step
// loads src[i-2:i+17) and stores dst[p:p+16).
func steps16(dst, src []byte, p, i, end int) (int, int) {
	if len(src) < 19 {
		return p, i
	}
	eq := archsimd.BroadcastUint8x16('=')
	cr := archsimd.BroadcastUint8x16('\r')
	lf := archsimd.BroadcastUint8x16('\n')
	dot := archsimd.BroadcastUint8x16('.')
	ky := archsimd.BroadcastUint8x16('y')
	k42 := archsimd.BroadcastUint8x16(42)
	k64 := archsimd.BroadcastUint8x16(64)
	crlf := archsimd.LoadUint8x16Array(&neonCRLF)
	wt := archsimd.LoadUint8x16Array(&neonWeights)
	sq := unsafe.Add(unsafe.Pointer(unsafe.SliceData(src)), i) // &src[i]
	dq := unsafe.Add(unsafe.Pointer(unsafe.SliceData(dst)), p) // &dst[p]
	stop := unsafe.Add(unsafe.Pointer(unsafe.SliceData(src)), min(end, len(src)-16))
	for uintptr(sq) < uintptr(stop) {
		v := archsimd.LoadUint8x16Array(at16(sq, 0))
		b1 := archsimd.LoadUint8x16Array(at16(sq, -1)) // the byte before each byte
		b2 := archsimd.LoadUint8x16Array(at16(sq, -2)) // and the one before that
		a1 := archsimd.LoadUint8x16Array(at16(sq, 1))  // the byte after
		e := v8(v.Equal(eq))
		c := crlf.LookupOrZero(v).Or(e) // '=', '\r', '\n' are never output
		s := v8(b1.Equal(eq))           // escaped by the byte before
		// right after "\r\n", a '=' or '.'
		ld := v8(b1.Equal(lf)).And(v8(b2.Equal(cr))).And(e.Or(v8(v.Equal(dot))))
		// "==", an escaped \r or \n; "\r\n=y", "\r\n.\r", "\r\n.=" (end
		// sequences; "\r\n.y", "\r\n=\r", "\r\n==" too, needlessly but
		// harmlessly)
		bad := s.And(c).Or(ld.And(v8(a1.Equal(ky)).Or(v8(a1.Equal(cr))).Or(v8(a1.Equal(eq)))))
		if nonzero(bad) {
			break
		}
		o := s.MulAdd(k64, v.Sub(k42))                               // escaped bytes lose 42+64, the others 42
		m := uint32(c.Or(ld).And(wt).ReshapeToUint16s().ReduceSum()) // the drop mask, see neonWeights
		o.LookupOrZero(archsimd.LoadUint8x16Array(&neonPack.idx[m&0x7fff])).StoreArray((*[16]uint8)(dq))
		dq = unsafe.Add(dq, neonPack.cnt[m&0xffff])
		sq = unsafe.Add(sq, 16)
	}
	return int(uintptr(dq) - uintptr(unsafe.Pointer(unsafe.SliceData(dst)))),
		int(uintptr(sq) - uintptr(unsafe.Pointer(unsafe.SliceData(src))))
}
