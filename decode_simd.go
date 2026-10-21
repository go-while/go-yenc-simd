//go:build goexperiment.simd && (amd64 || arm64)

package yencsimd

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
