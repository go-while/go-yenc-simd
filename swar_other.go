//go:build !amd64

package yencsimd

// swarWords returns the words decodePlain builds its constants from. It is not
// inlined, so they are values the compiler keeps in registers: on arm64 a
// 64-bit constant is four instructions, and the compiler rebuilt six of them
// in every iteration of the word loop (27% of it).
//
//go:noinline
func swarWords() (ones, highs uint64) { return 0x0101010101010101, 0x8080808080808080 }
