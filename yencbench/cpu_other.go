//go:build !linux && !darwin && !dragonfly && !freebsd && !netbsd && !openbsd && !windows

package yencbench

func cpuName() string { return "" }
