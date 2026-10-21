package yencbench

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// parseCPUInfo returns the CPU model from the text of Linux's /proc/cpuinfo:
// the "model name" (x86 and others), or on arm64, which has none, the cores
// that "CPU implementer" and "CPU part" name: "Neoverse-N2", or both kinds
// of a big.LITTLE CPU, "Cortex-A55 + Cortex-A76".
func parseCPUInfo(s string) string {
	var cores []string
	impl := ""
	for line := range strings.Lines(s) {
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		switch v = strings.TrimSpace(v); strings.TrimSpace(k) {
		case "model name":
			return strings.Join(strings.Fields(v), " ")
		case "CPU implementer":
			impl = v
		case "CPU part":
			if c := armCore(impl, v); !slices.Contains(cores, c) {
				cores = append(cores, c)
			}
		}
	}
	return strings.Join(cores, " + ")
}

// armCore names an arm64 core by its implementer and part number, as
// /proc/cpuinfo prints them ("0x41", "0xd49"), with the names of Linux's
// arch/arm64/include/asm/cputype.h. Other parts are given by number.
func armCore(impl, part string) string {
	i, err1 := strconv.ParseUint(impl, 0, 8)
	p, err2 := strconv.ParseUint(part, 0, 12)
	switch {
	case err1 != nil || err2 != nil:
		return "CPU implementer " + impl + " part " + part
	case armCores[i<<12|p] != "":
		return armCores[i<<12|p]
	case armImplementers[i] != "":
		return fmt.Sprintf("%s part %#x", armImplementers[i], p)
	}
	return fmt.Sprintf("CPU implementer %#x part %#x", i, p)
}

var armImplementers = map[uint64]string{
	0x41: "Arm", 0x42: "Broadcom", 0x43: "Cavium", 0x46: "Fujitsu", 0x48: "HiSilicon",
	0x4e: "NVIDIA", 0x50: "APM", 0x51: "Qualcomm", 0x61: "Apple", 0x6d: "Microsoft", 0xc0: "Ampere",
}

// armCores are the cores of most arm64 servers, boards and laptops, by
// implementer<<12 | part. Graviton 2, 3 and 4 are Neoverse-N1, -V1 and
// -V2, Ampere Altra is Neoverse-N1, a Raspberry Pi 4 or 5 Cortex-A72 or -A76.
var armCores = map[uint64]string{
	0x41<<12 | 0xd03: "Cortex-A53",
	0x41<<12 | 0xd04: "Cortex-A35",
	0x41<<12 | 0xd05: "Cortex-A55",
	0x41<<12 | 0xd07: "Cortex-A57",
	0x41<<12 | 0xd08: "Cortex-A72",
	0x41<<12 | 0xd09: "Cortex-A73",
	0x41<<12 | 0xd0a: "Cortex-A75",
	0x41<<12 | 0xd0b: "Cortex-A76",
	0x41<<12 | 0xd0c: "Neoverse-N1",
	0x41<<12 | 0xd0d: "Cortex-A77",
	0x41<<12 | 0xd40: "Neoverse-V1",
	0x41<<12 | 0xd41: "Cortex-A78",
	0x41<<12 | 0xd44: "Cortex-X1",
	0x41<<12 | 0xd46: "Cortex-A510",
	0x41<<12 | 0xd47: "Cortex-A710",
	0x41<<12 | 0xd48: "Cortex-X2",
	0x41<<12 | 0xd49: "Neoverse-N2",
	0x41<<12 | 0xd4b: "Cortex-A78C",
	0x41<<12 | 0xd4d: "Cortex-A715",
	0x41<<12 | 0xd4e: "Cortex-X3",
	0x41<<12 | 0xd4f: "Neoverse-V2",
	0x41<<12 | 0xd80: "Cortex-A520",
	0x41<<12 | 0xd81: "Cortex-A720",
	0x41<<12 | 0xd82: "Cortex-X4",
	0x41<<12 | 0xd84: "Neoverse-V3",
	0x41<<12 | 0xd85: "Cortex-X925",
	0x41<<12 | 0xd87: "Cortex-A725",
	0x41<<12 | 0xd8e: "Neoverse-N3",
	0x46<<12 | 0x001: "A64FX",
	0x48<<12 | 0xd01: "TSV110 (Kunpeng 920)",
	0x4e<<12 | 0x004: "Carmel",
	0x51<<12 | 0x001: "Oryon",
	0x6d<<12 | 0xd49: "Azure Cobalt 100",
	0xc0<<12 | 0xac3: "Ampere-1",
	0xc0<<12 | 0xac4: "Ampere-1a",
}
