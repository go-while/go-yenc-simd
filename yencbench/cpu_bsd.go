//go:build darwin || dragonfly || freebsd || netbsd || openbsd

package yencbench

import (
	"runtime"
	"strings"
	"syscall"
)

// cpuName is the CPU model from sysctl: machdep.cpu.brand_string on macOS
// ("Apple M2 Pro"), hw.model on the other BSDs; or "".
func cpuName() string {
	key := "hw.model"
	if runtime.GOOS == "darwin" || runtime.GOOS == "ios" {
		key = "machdep.cpu.brand_string"
	}
	s, err := syscall.Sysctl(key)
	if err != nil {
		return ""
	}
	return strings.Join(strings.Fields(s), " ")
}
