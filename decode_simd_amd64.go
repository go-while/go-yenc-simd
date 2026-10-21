//go:build goexperiment.simd && amd64

package yencsimd

import (
	"errors"
	"math/bits"
	"simd/archsimd"
	"sync"
	"unsafe"
)

var (
	hasVBMI2 = archsimd.X86.AVX512() && archsimd.X86.AVX512VBMI() && archsimd.X86.AVX512VBMI2() && archsimd.X86.AVX512VPOPCNTDQ() // VPCOMPRESSB packs 64 bytes, VPERMI2B shifts them, VPOPCNTQ counts them
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

// cpu* remember what the CPU can do; SetKernel changes what is used.
var cpuVBMI2, cpuAVX2, cpuAVX = hasVBMI2, hasAVX2, hasAVX

// Kernels lists the decode paths this CPU can run, fastest first.
func Kernels() []string {
	var k []string
	if cpuVBMI2 {
		k = append(k, "avx512vbmi2")
	}
	if cpuAVX2 {
		k = append(k, "avx2")
	}
	if cpuAVX {
		k = append(k, "avx")
	}
	return append(k, "scalar")
}

// SetKernel limits the decoder to the named path, one of Kernels(). It is
// meant for benchmarks and tests: it changes package state, so call it only
// while nothing is decoding.
func SetKernel(name string) error {
	switch name {
	case "avx512vbmi2":
		if cpuVBMI2 {
			hasVBMI2, hasAVX2, hasAVX = true, cpuAVX2, cpuAVX
			return nil
		}
	case "avx2":
		if cpuAVX2 {
			hasVBMI2, hasAVX2, hasAVX = false, true, cpuAVX
			return nil
		}
	case "avx":
		if cpuAVX {
			hasVBMI2, hasAVX2, hasAVX = false, false, true
			return nil
		}
	case "scalar":
		hasVBMI2, hasAVX2, hasAVX = false, false, false
		return nil
	}
	return errors.New("yencsimd: kernel " + name + " is not available on this CPU")
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
// decodes 64 bytes per step (AVX2, steps32), then 32 or 16 (AVX, steps16)
// near the end of src: plain bytes, escapes ('=' + byte), line breaks and
// stuffed dots ("\r\n." drops the dot). A step without a '=', '\r' or '\n'
// is just stored minus 42 (the one branch on the data); the others work out
// escapes and line starts from loads one and two bytes back, so a block may
// start or end anywhere, even inside "\r\n" or after '='. A block with
// anything the vector code doesn't do ends the loops: "==" (which '='
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
	packOnce.Do(fillPack)
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
			archsimd.ClearAVXUpperBits() // VZEROUPPER: the SSE code after us (crc32) must not pay for dirty upper halves
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

// pack holds the tables of the AVX2 and AVX loops in one struct, so that one
// register addresses both: 576 KiB, filled by fillPack before the first
// decode (0.2 ms on a 2 GHz Comet Lake, 0.06 ms on a Zen 5; a CPU with
// AVX-512 VBMI2 never needs it).
//
// idx[m] are VPSHUFB indices that pack the bytes NOT in the low 15 bits of
// drop mask m (bit set = dropped) of a 16-byte lane to the front, byte 15
// right after them: whether byte 15 is dropped only changes the count, not
// where the others go, so the table has 2^15 entries, not 2^16. The bytes
// after are garbage (stale indices, 16..30: VPSHUFB uses their low 4 bits),
// overwritten by the next store.
//
// cnt[m] is the number of bits clear in the 16-bit mask m: the bytes a lane
// keeps. A table, not bits.OnesCount: its fallback call (for CPUs without
// POPCNT) made the compiler reload every vector constant in the loop.
var pack struct {
	idx [1 << 15][16]int8
	cnt [1 << 16]uint8
}

var packOnce sync.Once

// fillPack fills pack. idx[m] is byte 0 if m keeps it, then idx[m>>1] with
// every index one higher: the kept bytes of m>>1 are those of m moved down
// one, and its trailing 15 becomes a 16 at a garbage position.
func fillPack() {
	const ones = 0x0101010101010101
	for k := range 16 {
		pack.idx[0][k] = int8(k)
	}
	for m := 1; m < 1<<15; m++ {
		src := (*[2]uint64)(unsafe.Pointer(&pack.idx[m>>1]))
		lo, hi := src[0]+ones, src[1]+ones
		if m&1 == 0 {
			lo, hi = lo<<8, hi<<8|lo>>56
		}
		dst := (*[2]uint64)(unsafe.Pointer(&pack.idx[m]))
		dst[0], dst[1] = lo, hi
	}
	pack.cnt[0] = 16
	for m := 1; m < 1<<16; m++ {
		pack.cnt[m] = pack.cnt[m>>1] - uint8(m&1)
	}
}

// nibYCR is a VPSHUFB table on the low nibble of a byte that gives 'y' or
// '\r' where the byte can be one: x == nibYCR[x&15] is x in {'y', '\r'}.
// (0x80 never equals a byte with the high bit clear, and VPSHUFB gives 0 for
// a byte with it set.)
var nibYCR = func() (t [32]uint8) {
	for i := range t {
		t[i] = 0x80
	}
	for _, c := range []byte{'y', '\r'} {
		t[c&15], t[16+c&15] = c, c
	}
	return
}()

// nibCRLF is the same for '\r' and '\n': x == nibCRLF[x&15] is x in
// {'\r', '\n'}, one instruction less than two compares. steps16 uses it;
// in steps32 the register it takes made the compiler spill, which cost more.
var nibCRLF = func() (t [32]uint8) {
	for i := range t {
		t[i] = 0x80
	}
	for _, c := range []byte{'\r', '\n'} {
		t[c&15], t[16+c&15] = c, c
	}
	return
}()

// steps32 decodes 64 bytes per step from src[i] with AVX2 while 72 bytes are
// left and no block needs the state machine (see decodeSIMD). The caller
// guarantees the state machine invariant at i.
//
// A step is two 32-byte vectors. If neither has a '=', '\r' or '\n', and the
// byte before the step is plain (notPlain, from the step before: a step that
// ends in a special byte, or in a stuffed dot, which is harmless), the output
// is the 64 bytes minus 42. Otherwise each vector is packed with two shuffles
// from pack, one per 16 bytes, each followed by a 16-byte store at the output
// position so far; the second store overwrites the garbage of the first. With
// random data a step in five takes the fast path, so the branch is
// mispredicted that often; with text almost every step does.
//
// Bounds (unchecked loads and stores; i+72 <= n = len(src) <= len(dst),
// p <= i, i >= 2): the loads are src[i-2:i+65), the stores reach dst[p+64).
func steps32(dst, src []byte, p, i int) (int, int) {
	if len(src) < 72 {
		return p, i
	}
	eq := archsimd.BroadcastUint8x32('=')
	cr := archsimd.BroadcastUint8x32('\r')
	lf := archsimd.BroadcastUint8x32('\n')
	dot := archsimd.BroadcastUint8x32('.')
	k42 := archsimd.BroadcastUint8x32(42)
	k64 := archsimd.BroadcastUint8x32(64)
	yCR := archsimd.LoadUint8x32Array(&nibYCR)
	idx8 := (*[1 << 16][8]int8)(unsafe.Pointer(&pack.idx)) // pack.idx in 8-byte halves: an index the CPU can scale
	idx := func(m uint32) archsimd.Int8x16 {               // pack.idx[m&0x7fff]
		return archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Pointer(&idx8[(m&0x7fff)<<1])))
	}
	cnt := func(m uint32) int { return int(pack.cnt[m&0xffff]) }
	sq := unsafe.Add(unsafe.Pointer(unsafe.SliceData(src)), i) // &src[i]
	dq := unsafe.Add(unsafe.Pointer(unsafe.SliceData(dst)), p) // &dst[p]
	last := unsafe.Add(unsafe.Pointer(unsafe.SliceData(src)), len(src)-72)
	notPlain := uint32(0) // the caller guarantees a plain byte before src[i]
	for uintptr(sq) <= uintptr(last) {
		v0 := archsimd.LoadUint8x32Array(at32(sq, 0))
		v1 := archsimd.LoadUint8x32Array(at32(sq, 32))
		ctl0 := v0.Equal(eq).Or(v0.Equal(cr)).Or(v0.Equal(lf)) // '=', '\r', '\n' are never output
		ctl1 := v1.Equal(eq).Or(v1.Equal(cr)).Or(v1.Equal(lf))
		if ctl0.Or(ctl1).ToBits()|notPlain == 0 {
			v0.Sub(k42).StoreArray((*[32]uint8)(dq))
			v1.Sub(k42).StoreArray((*[32]uint8)(unsafe.Add(dq, 32)))
			dq = unsafe.Add(dq, 64)
		} else {
			// the first vector
			b1 := archsimd.LoadUint8x32Array(at32(sq, -1)) // the byte before each byte
			b2 := archsimd.LoadUint8x32Array(at32(sq, -2)) // and the one before that
			a1 := archsimd.LoadUint8x32Array(at32(sq, 1))  // the byte after
			esc := b1.Equal(eq)                            // escaped by the byte before
			// right after "\r\n", a '=' or '.'
			ld := b1.Equal(lf).And(b2.Equal(cr)).And(v0.Equal(eq).Or(v0.Equal(dot)))
			// "==" (the second '=' would be escaped), an escaped \r or \n;
			// "\r\n=y", "\r\n.\r" and "\r\n.=" (end sequences: a line-start
			// '=' or '.' before 'y', '\r' or '='; "\r\n.y", "\r\n=\r" and
			// "\r\n==" are flagged too, needlessly but harmlessly)
			bad := esc.And(ctl0).Or(ld.And(a1.Equal(yCR.PermuteOrZeroGrouped(a1.AsInt8x32())).Or(a1.Equal(eq))))
			if !bad.ToInt8x32().IsZero() {
				break
			}
			drop := ctl0.Or(ld).ToBits() // not output: ctl and the stuffed dot
			// escaped bytes lose 42+64, the others 42
			out := v0.Sub(k42).Sub(k64.Masked(esc))
			out.GetLo().PermuteOrZero(idx(drop)).StoreArray((*[16]uint8)(dq))
			dq = unsafe.Add(dq, cnt(drop))
			drop >>= 16
			out.GetHi().PermuteOrZero(idx(drop)).StoreArray((*[16]uint8)(dq))
			dq = unsafe.Add(dq, cnt(drop))
			// the second vector, the same (loaded again: a register less to keep)
			v1 = archsimd.LoadUint8x32Array(at32(sq, 32))
			b1 = archsimd.LoadUint8x32Array(at32(sq, 31))
			b2 = archsimd.LoadUint8x32Array(at32(sq, 30))
			a1 = archsimd.LoadUint8x32Array(at32(sq, 33))
			esc = b1.Equal(eq)
			ld = b1.Equal(lf).And(b2.Equal(cr)).And(v1.Equal(eq).Or(v1.Equal(dot)))
			bad = esc.And(ctl1).Or(ld.And(a1.Equal(yCR.PermuteOrZeroGrouped(a1.AsInt8x32())).Or(a1.Equal(eq))))
			if !bad.ToInt8x32().IsZero() {
				sq = unsafe.Add(sq, 32) // the first vector is done
				break
			}
			drop = ctl1.Or(ld).ToBits()
			out = v1.Sub(k42).Sub(k64.Masked(esc))
			out.GetLo().PermuteOrZero(idx(drop)).StoreArray((*[16]uint8)(dq))
			dq = unsafe.Add(dq, cnt(drop))
			notPlain = drop >> 31 // the last byte was dropped: special (or a stuffed dot)
			drop >>= 16
			out.GetHi().PermuteOrZero(idx(drop)).StoreArray((*[16]uint8)(dq))
			dq = unsafe.Add(dq, cnt(drop))
		}
		sq = unsafe.Add(sq, 64)
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
			last := unsafe.Add(sbase, n-64) // where the last block that fits starts
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
				if uintptr(sp)+64 > uintptr(last) {
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
			archsimd.ClearAVXUpperBits() // VZEROUPPER: the SSE code after us (crc32) must not pay for dirty upper halves
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
	if i <= s || i >= n {
		return p, i // outside its range: the state machine finishes
	}
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

// splat16 are the constants of steps16, loaded with VMOVDQU (AVX):
// archsimd.BroadcastUint8x16 is VPBROADCASTB, an AVX2 instruction, and
// steps16 runs on CPUs with AVX but not AVX2 (TestKernelISA checks).
var splat16 = struct{ eq, cr, lf, dot, k42, k64 [16]uint8 }{
	fill16('='), fill16('\r'), fill16('\n'), fill16('.'), fill16(42), fill16(64),
}

func fill16(c byte) (t [16]uint8) {
	for i := range t {
		t[i] = c
	}
	return
}

// steps16 is steps32 with 16-byte vectors (AVX): from src[i] while at least
// 24 bytes are left and no block needs the state machine. It also decodes
// the tail of src after steps32, so a step is one vector, and the fast path
// takes two while 48 bytes are left. The caller guarantees the state machine
// invariant at i (see decodeSIMD), but src[i-1] may be a '=' or "\r\n"
// (where steps32 stopped): notPlain starts from it.
//
// Bounds (unchecked; i+24 <= len(src) <= len(dst), p <= i, i >= 2): a step
// loads src[i-2:i+17) and stores dst[p:p+16); the fast path loads
// src[i:i+32) and stores dst[p:p+32), with i+48 <= len(src).
func steps16(dst, src []byte, p, i int) (int, int) {
	if len(src) < 24 {
		return p, i
	}
	eq := archsimd.LoadUint8x16Array(&splat16.eq)
	cr := archsimd.LoadUint8x16Array(&splat16.cr)
	lf := archsimd.LoadUint8x16Array(&splat16.lf)
	dot := archsimd.LoadUint8x16Array(&splat16.dot)
	k42 := archsimd.LoadUint8x16Array(&splat16.k42)
	k64 := archsimd.LoadUint8x16Array(&splat16.k64)
	yCR := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(&nibYCR)))
	crlf := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(&nibCRLF)))
	idx8 := (*[1 << 16][8]int8)(unsafe.Pointer(&pack.idx)) // pack.idx in 8-byte halves: an index the CPU can scale
	idx := func(m uint32) archsimd.Int8x16 {               // pack.idx[m&0x7fff]
		return archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Pointer(&idx8[(m&0x7fff)<<1])))
	}
	cnt := func(m uint32) int { return int(pack.cnt[m&0xffff]) }
	sq := unsafe.Add(unsafe.Pointer(unsafe.SliceData(src)), i) // &src[i]
	dq := unsafe.Add(unsafe.Pointer(unsafe.SliceData(dst)), p) // &dst[p]
	last := unsafe.Add(unsafe.Pointer(unsafe.SliceData(src)), len(src)-24)
	var pair unsafe.Pointer // where a second vector still fits; nil (never reached) if none does
	if len(src) >= 48 {
		pair = unsafe.Add(unsafe.Pointer(unsafe.SliceData(src)), len(src)-48)
	}
	// the context at src[i]: steps32 may have stopped after a '=' or "\r\n"
	notPlain := uint32(0)
	if c := src[i-1]; c == '=' || c == '\n' && src[i-2] == '\r' {
		notPlain = 1
	}
	for uintptr(sq) <= uintptr(last) {
		v0 := archsimd.LoadUint8x16Array(at16(sq, 0))
		ctl0 := v0.Equal(eq).Or(v0.Equal(crlf.PermuteOrZero(v0.AsInt8x16()))) // '=', '\r', '\n' are never output
		two := uintptr(sq) <= uintptr(pair)                                   // a second vector fits
		var v1 archsimd.Uint8x16
		var ctl1 archsimd.Mask8x16
		fast := false
		if two {
			v1 = archsimd.LoadUint8x16Array(at16(sq, 16))
			ctl1 = v1.Equal(eq).Or(v1.Equal(crlf.PermuteOrZero(v1.AsInt8x16())))
			fast = uint32(ctl0.Or(ctl1).ToBits())|notPlain == 0
		}
		if fast {
			v0.Sub(k42).StoreArray((*[16]uint8)(dq))
			v1.Sub(k42).StoreArray((*[16]uint8)(unsafe.Add(dq, 16)))
			dq = unsafe.Add(dq, 32)
			sq = unsafe.Add(sq, 32)
			continue
		}
		// the first vector
		b1 := archsimd.LoadUint8x16Array(at16(sq, -1)) // the byte before each byte
		b2 := archsimd.LoadUint8x16Array(at16(sq, -2)) // and the one before that
		a1 := archsimd.LoadUint8x16Array(at16(sq, 1))  // the byte after
		esc := b1.Equal(eq)                            // escaped by the byte before
		// right after "\r\n", a '=' or '.'
		ld := b1.Equal(lf).And(b2.Equal(cr)).And(v0.Equal(eq).Or(v0.Equal(dot)))
		// the bad bytes, as in steps32
		bad := esc.And(ctl0).Or(ld.And(a1.Equal(yCR.PermuteOrZero(a1.AsInt8x16())).Or(a1.Equal(eq))))
		if !bad.ToInt8x16().IsZero() {
			break
		}
		drop := uint32(ctl0.Or(ld).ToBits()) // not output: ctl and the stuffed dot
		// escaped bytes lose 42+64, the others 42
		out := v0.Sub(k42).Sub(k64.Masked(esc))
		out.PermuteOrZero(idx(drop)).StoreArray((*[16]uint8)(dq))
		dq = unsafe.Add(dq, cnt(drop))
		notPlain = drop >> 15 // the last byte was dropped: special (or a stuffed dot)
		sq = unsafe.Add(sq, 16)
		if !two {
			continue
		}
		// the second vector, the same
		b1 = archsimd.LoadUint8x16Array(at16(sq, -1))
		b2 = archsimd.LoadUint8x16Array(at16(sq, -2))
		a1 = archsimd.LoadUint8x16Array(at16(sq, 1))
		esc = b1.Equal(eq)
		ld = b1.Equal(lf).And(b2.Equal(cr)).And(v1.Equal(eq).Or(v1.Equal(dot)))
		bad = esc.And(ctl1).Or(ld.And(a1.Equal(yCR.PermuteOrZero(a1.AsInt8x16())).Or(a1.Equal(eq))))
		if !bad.ToInt8x16().IsZero() {
			break
		}
		drop = uint32(ctl1.Or(ld).ToBits())
		out = v1.Sub(k42).Sub(k64.Masked(esc))
		out.PermuteOrZero(idx(drop)).StoreArray((*[16]uint8)(dq))
		dq = unsafe.Add(dq, cnt(drop))
		notPlain = drop >> 15
		sq = unsafe.Add(sq, 16)
	}
	return int(uintptr(dq) - uintptr(unsafe.Pointer(unsafe.SliceData(dst)))),
		int(uintptr(sq) - uintptr(unsafe.Pointer(unsafe.SliceData(src))))
}
