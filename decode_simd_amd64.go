//go:build goexperiment.simd && amd64

package yencsimd

import (
	"math/bits"
	"simd/archsimd"
	"unsafe"
)

var (
	hasVBMI2 = archsimd.X86.AVX512VBMI2() && archsimd.X86.AVX512VPOPCNTDQ() // VPCOMPRESSB packs 64 bytes in one instruction; VPOPCNTQ counts them
	hasAVX2  = archsimd.X86.AVX2()
	hasAVX   = archsimd.X86.AVX()
)

// Kernel reports which fast path the decoder uses.
func Kernel() string {
	switch {
	case hasVBMI2:
		return "avx512vbmi2"
	case hasAVX2:
		return "avx2"
	case hasAVX:
		return "avx"
	}
	return "scalar"
}

// lut holds the tables of the vector loops in one struct, so that one
// register addresses all of them.
//
// lo[m] / hi[m] are VPSHUFB indices that pack the bytes kept by mask m (bit
// set = keep) of the low / high 8 bytes of a vector to the front. An entry is
// 8 bytes, so its address is base + m*8 (one LEA); VPSHUFB reads 16, the
// other 8 are the next entry and shuffle garbage into the output bytes 8-15,
// which a later store overwrites. hi[255] reads into count.
//
// count[m] is the number of bits set in m. A table, not bits.OnesCount: its
// fallback call (for CPUs without POPCNT) made the compiler reload every
// vector constant in the loop.
var lut struct {
	lo, hi [256][8]int8
	count  [256]uint8
}

func init() {
	for m := range 256 {
		lut.count[m] = uint8(bits.OnesCount8(uint8(m)))
		k := 0
		for b := range 8 {
			if m&(1<<b) != 0 {
				lut.lo[m][k] = int8(b)
				lut.hi[m][k] = int8(b + 8)
				k++
			}
		}
		for ; k < 8; k++ {
			lut.lo[m][k] = -1
			lut.hi[m][k] = -1
		}
	}
}

// shuffle returns the lo / hi (off = 0 / 1) index vector for mask m.
func shuffle(off int, m uint32) archsimd.Int8x16 {
	return archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Add(unsafe.Pointer(&lut), (uintptr(off)<<11)+uintptr(m)<<3)))
}

func decode(dst, src []byte, state *State) (int, int, End) {
	switch {
	case hasVBMI2:
		return decodeVBMI2(dst, src, state)
	case hasAVX:
		return decodeSIMD(dst, src, state)
	}
	return decodeScalar(dst, src, state)
}

// decodeSIMD is the state machine with vector loops. Once in StateNone it
// decodes 32 bytes per step (AVX2, steps32), then 16 (AVX, steps16) near the
// end of src, without branching on the data: plain bytes, escapes ('=' +
// byte), line breaks and stuffed dots ("\r\n." drops the dot). Each step
// works out escapes and line starts from loads one and two bytes back, so a
// block may start or end anywhere, even inside "\r\n" or after '='. A block
// with anything the vector code doesn't do ends the loops: "==" (which '='
// escapes which?), an escaped \r or \n (invalid yEnc; rapidyenc has quirks
// for it) and the end sequences "\r\n=y", "\r\n.\r", "\r\n.=". The state
// machine takes it from there, one byte at a time.
//
// Stores write whole vectors at dst[p:]; the bytes past the output are
// overwritten later. That needs len(dst) >= len(src) and p <= i (see the
// loops for their bounds).
//
// The loops are separate functions, called once per run of blocks: a loop
// in a function that also calls the state machine kept its vector constants
// on the stack, and a bounds check or a load of hasAVX2 per block costs
// more than the call.
func decodeSIMD(dst, src []byte, state *State) (int, int, End) {
	st := *state
	p, i, n := 0, 0, len(src)
	for i < n {
		// The vector loops need the bytes before i to mean what they say: in
		// StateNone src[i-1] can't be an unescaped '\r' or end "\r\n", but it
		// can be an escaped '='.
		if st == StateNone && i >= 2 && i+24 <= n && src[i-1] != '=' {
			start := i
			if hasAVX2 {
				p, i = steps32(dst, src, p, i)
			}
			p, i = steps16(dst, src, p, i)
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

// steps32 decodes 32 bytes per step from src[i] with AVX2 while 64 bytes are
// left and no block needs the state machine (see decodeSIMD). The caller
// guarantees the state machine invariant at i.
//
// A block is packed with four shuffles from lut, one per 8 bytes, each
// followed by a 16-byte store at the output position so far; the stores
// after the first overwrite the garbage of the one before. There is no
// branch on the data (a branch on the number of special bytes in a block is
// mispredicted for one block in five with random data, and the flush costs
// more than the shuffles save).
//
// Bounds (unchecked loads and stores; i+64 <= n = len(src) <= len(dst),
// p <= i, i >= 2): the loads are src[i-2:i+33), the stores reach dst[p+40).
func steps32(dst, src []byte, p, i int) (int, int) {
	if len(src) < 64 {
		return p, i
	}
	eq := archsimd.BroadcastUint8x32('=')
	cr := archsimd.BroadcastUint8x32('\r')
	lf := archsimd.BroadcastUint8x32('\n')
	dot := archsimd.BroadcastUint8x32('.')
	ky := archsimd.BroadcastUint8x32('y')
	k42 := archsimd.BroadcastUint8x32(42)
	k64 := archsimd.BroadcastUint8x32(64)
	load := func(q unsafe.Pointer, off int) archsimd.Uint8x32 {
		return archsimd.LoadUint8x32Array((*[32]uint8)(unsafe.Add(q, off)))
	}
	tab := unsafe.Pointer(&lut)
	shuf := func(off uintptr, m uint32) archsimd.Int8x16 { // lut.lo[m] (off 0) or lut.hi[m]
		return archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Add(tab, uintptr(m)<<3+off)))
	}
	count := func(m uint32) int { return int(*(*uint8)(unsafe.Add(tab, unsafe.Offsetof(lut.count)+uintptr(m)))) }
	sq := unsafe.Add(unsafe.Pointer(unsafe.SliceData(src)), i) // &src[i]
	dq := unsafe.Add(unsafe.Pointer(unsafe.SliceData(dst)), p) // &dst[p]
	last := unsafe.Add(unsafe.Pointer(unsafe.SliceData(src)), len(src)-64)
	for uintptr(sq) <= uintptr(last) {
		v := load(sq, 0)
		b1 := load(sq, -1) // the byte before each byte
		b2 := load(sq, -2) // and the one before that
		a1 := load(sq, 1)  // the byte after
		isEq, isDot := v.Equal(eq), v.Equal(dot)
		ctl := isEq.Or(v.Equal(cr)).Or(v.Equal(lf)) // '=', '\r', '\n' are never output
		esc := b1.Equal(eq)                         // escaped by the byte before
		line := b1.Equal(lf).And(b2.Equal(cr))      // right after "\r\n"
		// "==" (the second '=' would be escaped), an escaped \r or \n,
		// "\r\n=y", "\r\n.\r" and "\r\n.="
		bad := esc.And(ctl).
			Or(line.And(isEq).And(a1.Equal(ky))).
			Or(line.And(isDot).And(a1.Equal(cr).Or(a1.Equal(eq))))
		if !bad.ToInt8x32().IsZero() {
			break
		}
		keep := ^ctl.Or(line.And(isDot)).ToBits()
		// escaped bytes lose 42+64, the others 42
		out := v.Sub(k42).Sub(k64.Masked(esc))
		lo, hi := out.GetLo(), out.GetHi()
		m0, m1, m2, m3 := keep&0xff, keep>>8&0xff, keep>>16&0xff, keep>>24
		lo.PermuteOrZero(shuf(0, m0)).StoreArray((*[16]uint8)(dq))
		dq = unsafe.Add(dq, count(m0))
		lo.PermuteOrZero(shuf(unsafe.Offsetof(lut.hi), m1)).StoreArray((*[16]uint8)(dq))
		dq = unsafe.Add(dq, count(m1))
		hi.PermuteOrZero(shuf(0, m2)).StoreArray((*[16]uint8)(dq))
		dq = unsafe.Add(dq, count(m2))
		hi.PermuteOrZero(shuf(unsafe.Offsetof(lut.hi), m3)).StoreArray((*[16]uint8)(dq))
		dq = unsafe.Add(dq, count(m3))
		sq = unsafe.Add(sq, 32)
	}
	return int(uintptr(dq) - uintptr(unsafe.Pointer(unsafe.SliceData(dst)))),
		int(uintptr(sq) - uintptr(unsafe.Pointer(unsafe.SliceData(src))))
}

// shiftUp1 are VPERMI2B indices that move a vector up by one byte, filling
// byte 0 from the end of the vector before it: out[j] = (prev:cur)[63+j].
var shiftUp1 = func() (t [64]uint8) {
	for j := range t {
		t[j] = uint8(63 + j)
	}
	return
}()

// decodeVBMI2 is decodeSIMD for CPUs with AVX-512 VBMI2: 64 bytes per step,
// packed by VPCOMPRESSB in one instruction instead of shuffle tables. It is a
// separate function because a function that mentions 512-bit vectors at all
// uses 512-bit instructions (to zero them) on entry.
//
// The rules are vector arithmetic, not mask logic: x^c is zero where x is c,
// the Min of such vectors is zero where any is (or) and the Max where all
// are (and). Only two results become bitmaps, bad and keep. On Zen 5 the
// mask registers are the bottleneck: each compare into one, and each move
// from one to an integer register, costs about a cycle of the loop.
//
// A block with a bad byte is decoded up to that byte (VPCOMPRESSB with the
// mask cut there) and the state machine takes over at it, so a bail-out costs
// one block and a few bytes, not a byte loop over the block.
func decodeVBMI2(dst, src []byte, state *State) (int, int, End) {
	eq := archsimd.BroadcastUint8x64('=')
	cr := archsimd.BroadcastUint8x64('\r')
	lf := archsimd.BroadcastUint8x64('\n')
	dot := archsimd.BroadcastUint8x64('.')
	ky := archsimd.BroadcastUint8x64('y')
	one := archsimd.BroadcastUint8x64(1)
	k106 := archsimd.BroadcastUint8x64(42 + 64)
	ones := archsimd.BroadcastUint8x64(0xff)
	up1 := archsimd.LoadUint8x64Array(&shiftUp1)
	var zero archsimd.Uint8x64
	// The number of bytes kept is a VPOPCNTQ in the vector unit: the mask is
	// moved in and the count out. bits.OnesCount64 checks for POPCNT on
	// every call (GOAMD64=v1) and its fallback call made the compiler spill
	// around it; the count table costs 8 loads.
	var zero2 archsimd.Uint64x2

	st := *state
	p, i, n := 0, 0, len(src)
	for i < n {
		if st == StateNone && i >= 2 && i+72 <= n && src[i-1] != '=' {
			start := i
			// The loop walks with pointers, sp = &src[i] and dp = &dst[p].
			// unchecked: it loads src[i-2:i+64] with i >= 2 and i+64 <= n, and
			// stores dst[p:p+64] with p <= i, so p+64 <= n <= len(dst). The
			// pointers move on only when the next block fits, so they stay
			// inside src and dst.
			sbase := unsafe.Pointer(unsafe.SliceData(src))
			dbase := unsafe.Pointer(unsafe.SliceData(dst))
			sp := unsafe.Add(sbase, i)
			dp := unsafe.Add(dbase, p)
			last := uintptr(sbase) + uintptr(n-64) // where the last block that fits starts
			// lineX of the block before, for the first byte: no "\r\n=" or
			// "\r\n." is pending in StateNone (that would be StateCRLFEQ or
			// StateCRLFDT), so nothing is zero
			prevLineX := ones
			cnt := 0
			bailed := false
			for {
				v := archsimd.LoadUint8x64Array((*[64]uint8)(sp))
				b1 := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Add(sp, -1))) // the byte before each byte
				b2 := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Add(sp, -2))) // and the one before that
				xEq := v.Xor(eq)
				ctl := xEq.Min(v.Xor(cr)).Min(v.Xor(lf))                     // zero at '=', '\r', '\n': never output
				lineX := b2.Xor(cr).Max(b1.Xor(lf)).Max(xEq.Min(v.Xor(dot))) // zero at "\r\n=" and "\r\n."
				xA := prevLineX.ConcatPermute(lineX, up1)                    // zero where the byte before is that
				prevLineX = lineX
				esc := b1.Xor(eq) // zero where escaped by the byte before
				// zero at a bad byte: "==" (which '=' escapes which?) and an
				// escaped \r or \n (invalid; rapidyenc has quirks for it);
				// "\r\n=y" and "\r\n.\r", "\r\n.=" (end sequences). "\r\n.y"
				// is flagged too, needlessly but harmlessly: the state
				// machine agrees.
				badZ := esc.Max(ctl).Min(xA.Max(ctl.Min(v.Xor(ky))))
				keepK := ctl.Min(lineX).NotEqual(zero) // not output: ctl and the stuffed dot
				keep := keepK.ToBits()
				// escaped bytes lose 42+64, the others 42: min(esc, 1) is 1
				// where not escaped, so this is v-106+64 there (no mask)
				out := v.Sub(k106).Add(esc.Min(one).AsUint16x32().ShiftAllLeft(6).AsUint8x64())
				if bad := badZ.Equal(zero).ToBits(); bad != 0 {
					// decode up to the bad byte; the state machine takes it
					// from there (stateAfter sees '=', "\r\n." etc. before it)
					j := bits.TrailingZeros64(bad)
					keep &= 1<<uint(j) - 1
					out.Compress(archsimd.Mask8x64FromBits(keep)).StoreArray((*[64]uint8)(dp))
					cnt = int(zero2.SetElem(0, keep).OnesCount().GetElem(0))
					i = int(uintptr(sp)-uintptr(sbase)) + j
					bailed = true
					break
				}
				out.Compress(keepK).StoreArray((*[64]uint8)(dp))
				cnt = int(zero2.SetElem(0, keep).OnesCount().GetElem(0))
				if uintptr(sp)+64 > last {
					i = int(uintptr(sp)-uintptr(sbase)) + 64
					break
				}
				sp = unsafe.Add(sp, 64)
				dp = unsafe.Add(dp, cnt)
			}
			p = int(uintptr(dp)-uintptr(dbase)) + cnt
			if !bailed && i < n {
				// the last 1..63 bytes: one more block, overlapping the end
				p, i = tailVBMI2(dst, src, p, i)
			}
			if i != start {
				st = stateAfter(src, i)
				continue
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

// tailVBMI2 decodes the last 1..63 bytes of src, from i on, as one block that
// ends at len(src) (the rules are those of decodeVBMI2): the block overlaps
// the ov = 64-(len(src)-i) bytes before i, which are decoded already, so
// their bits are cleared from keep and bad. The bytes before each byte are in
// the block. It returns the new p and i: i < len(src) when a bad byte stopped
// it, the state machine takes it from there. Needs len(src) >= 66 and p <= i.
func tailVBMI2(dst, src []byte, p, i int) (int, int) {
	eq := archsimd.BroadcastUint8x64('=')
	cr := archsimd.BroadcastUint8x64('\r')
	lf := archsimd.BroadcastUint8x64('\n')
	dot := archsimd.BroadcastUint8x64('.')
	ky := archsimd.BroadcastUint8x64('y')
	one := archsimd.BroadcastUint8x64(1)
	k106 := archsimd.BroadcastUint8x64(42 + 64)
	ones := archsimd.BroadcastUint8x64(0xff)
	up1 := archsimd.LoadUint8x64Array(&shiftUp1)
	var zero archsimd.Uint8x64
	var zero2 archsimd.Uint64x2

	n := len(src)
	s := n - 64
	ov := uint(i - s)
	v := archsimd.LoadUint8x64Array((*[64]uint8)(src[s:]))
	b1 := archsimd.LoadUint8x64Array((*[64]uint8)(src[s-1:]))
	b2 := archsimd.LoadUint8x64Array((*[64]uint8)(src[s-2:]))
	xEq := v.Xor(eq)
	ctl := xEq.Min(v.Xor(cr)).Min(v.Xor(lf))
	lineX := b2.Xor(cr).Max(b1.Xor(lf)).Max(xEq.Min(v.Xor(dot)))
	xA := ones.ConcatPermute(lineX, up1) // byte ov-1 is in the block (ov >= 1)
	esc := b1.Xor(eq)
	badZ := esc.Max(ctl).Min(xA.Max(ctl.Min(v.Xor(ky))))
	fresh := ^uint64(0) << ov // the bytes not decoded yet
	bad := badZ.Equal(zero).ToBits() & fresh
	keep := ctl.Min(lineX).NotEqual(zero).ToBits() & fresh
	if bad != 0 {
		j := bits.TrailingZeros64(bad)
		keep &= 1<<uint(j) - 1
		i = s + j
	} else {
		i = n
	}
	out := v.Sub(k106).Add(esc.Min(one).AsUint16x32().ShiftAllLeft(6).AsUint8x64())
	cnt := int(zero2.SetElem(0, keep).OnesCount().GetElem(0))
	out.Compress(archsimd.Mask8x64FromBits(keep)).StorePart(dst[p : p+cnt]) // cnt <= 64-ov, the bytes from the old i = s+ov >= p
	return p + cnt, i
}

// steps16 is steps32 with 16-byte vectors (AVX): from src[i] while at least
// 24 bytes are left and no block needs the state machine. The caller
// guarantees the state machine invariant at i (see decodeSIMD).
//
// Bounds (unchecked; i+24 <= len(src) <= len(dst), p <= i, i >= 2): the
// loads are src[i-2:i+17), the stores reach dst[p+24).
func steps16(dst, src []byte, p, i int) (int, int) {
	if len(src) < 24 {
		return p, i
	}
	eq := archsimd.BroadcastUint8x16('=')
	cr := archsimd.BroadcastUint8x16('\r')
	lf := archsimd.BroadcastUint8x16('\n')
	dot := archsimd.BroadcastUint8x16('.')
	ky := archsimd.BroadcastUint8x16('y')
	k42 := archsimd.BroadcastUint8x16(42)
	k64 := archsimd.BroadcastUint8x16(64)
	load := func(q unsafe.Pointer, off int) archsimd.Uint8x16 {
		return archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(q, off)))
	}
	sq := unsafe.Add(unsafe.Pointer(unsafe.SliceData(src)), i) // &src[i]
	dq := unsafe.Add(unsafe.Pointer(unsafe.SliceData(dst)), p) // &dst[p]
	last := unsafe.Add(unsafe.Pointer(unsafe.SliceData(src)), len(src)-24)
	for uintptr(sq) <= uintptr(last) {
		v := load(sq, 0)
		b1 := load(sq, -1)
		b2 := load(sq, -2)
		a1 := load(sq, 1)
		isEq, isDot := v.Equal(eq), v.Equal(dot)
		ctl := isEq.Or(v.Equal(cr)).Or(v.Equal(lf))
		esc := b1.Equal(eq)
		line := b1.Equal(lf).And(b2.Equal(cr))
		bad := esc.And(ctl).
			Or(line.And(isEq).And(a1.Equal(ky))).
			Or(line.And(isDot).And(a1.Equal(cr).Or(a1.Equal(eq))))
		if !bad.ToInt8x16().IsZero() {
			break
		}
		keep := ^uint32(ctl.Or(line.And(isDot)).ToBits()) & 0xffff
		out := v.Sub(k42).Sub(k64.Masked(esc))
		out.PermuteOrZero(shuffle(0, keep&0xff)).StoreArray((*[16]uint8)(dq))
		dq = unsafe.Add(dq, int(lut.count[keep&0xff]))
		out.PermuteOrZero(shuffle(1, keep>>8)).StoreArray((*[16]uint8)(dq))
		dq = unsafe.Add(dq, int(lut.count[keep>>8]))
		sq = unsafe.Add(sq, 16)
	}
	return int(uintptr(dq) - uintptr(unsafe.Pointer(unsafe.SliceData(dst)))),
		int(uintptr(sq) - uintptr(unsafe.Pointer(unsafe.SliceData(src))))
}

// stateAfter is the state after the vector loop decoded up to src[i]: it
// follows from the last bytes, since the loop never decoded "==", an escaped
// line break or an end sequence.
func stateAfter(src []byte, i int) State {
	at := func(k int) byte { // src[i-k], or 0 before the start
		if i-k < 0 {
			return 0
		}
		return src[i-k]
	}
	lineStart := func(k int) bool { return at(k+1) == '\n' && at(k+2) == '\r' } // src[i-k] starts a line
	switch at(1) {
	case '=':
		if lineStart(1) || at(2) == '.' && lineStart(2) {
			return StateCRLFEQ
		}
		return StateEQ
	case '\r':
		if at(2) == '.' && lineStart(2) {
			return StateCRLFDTCR
		}
		return StateCR
	case '\n':
		if at(2) == '\r' {
			return StateCRLF
		}
	case '.':
		if lineStart(1) {
			return StateCRLFDT
		}
	}
	return StateNone
}
