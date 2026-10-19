package yencsimd

// swarWords returns the words decodePlain builds its constants from. On amd64
// it is inlined, so they stay constants: the compiler rebuilds them in every
// iteration, but a 64-bit constant is one instruction there.
func swarWords() (ones, highs uint64) { return 0x0101010101010101, 0x8080808080808080 }
