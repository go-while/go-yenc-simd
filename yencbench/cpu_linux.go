package yencbench

import "os"

// cpuName is the CPU model from /proc/cpuinfo (see parseCPUInfo), or "".
func cpuName() string {
	b, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return ""
	}
	return parseCPUInfo(string(b))
}
