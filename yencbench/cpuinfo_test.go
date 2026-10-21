package yencbench

import "testing"

func TestParseCPUInfo(t *testing.T) {
	for _, c := range []struct{ name, in, want string }{
		{"x86", "processor\t: 0\nvendor_id\t: AuthenticAMD\nmodel name\t: AMD EPYC 9B45 128-Core Processor\nflags\t\t: fpu\n",
			"AMD EPYC 9B45 128-Core Processor"},
		{"arm64", "processor\t: 0\nBogoMIPS\t: 2000.00\nFeatures\t: fp asimd\nCPU implementer\t: 0x41\nCPU architecture: 8\nCPU variant\t: 0x0\nCPU part\t: 0xd49\nCPU revision\t: 0\n\n" +
			"processor\t: 1\nCPU implementer\t: 0x41\nCPU part\t: 0xd49\n", "Neoverse-N2"},
		{"big.LITTLE", "CPU implementer\t: 0x41\nCPU part\t: 0xd05\nCPU implementer\t: 0x41\nCPU part\t: 0xd05\nCPU implementer\t: 0x41\nCPU part\t: 0xd0b\n",
			"Cortex-A55 + Cortex-A76"},
		{"unknown part", "CPU implementer\t: 0x61\nCPU part\t: 0x099\n", "Apple part 0x99"},
		{"unknown implementer", "CPU implementer\t: 0x99\nCPU part\t: 0x001\n", "CPU implementer 0x99 part 0x1"},
		{"empty", "", ""},
	} {
		if got := parseCPUInfo(c.in); got != c.want {
			t.Errorf("%s: got %q, want %q", c.name, got, c.want)
		}
	}
}
