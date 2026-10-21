//go:build goexperiment.simd && (amd64 || arm64)

package yencsimd

import "unsafe"

// at16 and at32 are the 16 and 32 bytes at q+off. The vector loops load
// through them, with the archsimd load at the call, not through a closure
// that loads: an inlined function whose code all comes from its own body
// leaves a NOP at each call (an inline mark, for tracebacks), and the loops
// would run those NOPs at every step.
func at16(q unsafe.Pointer, off int) *[16]uint8 { return (*[16]uint8)(unsafe.Add(q, off)) }
func at32(q unsafe.Pointer, off int) *[32]uint8 { return (*[32]uint8)(unsafe.Add(q, off)) }

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
