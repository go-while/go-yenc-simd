package yencbench

import (
	"strings"
	"syscall"
	"unsafe"
)

// cpuName is the CPU model from the registry (ProcessorNameString of the
// first processor), or "".
func cpuName() string {
	var k syscall.Handle
	path, _ := syscall.UTF16PtrFromString(`HARDWARE\DESCRIPTION\System\CentralProcessor\0`)
	if syscall.RegOpenKeyEx(syscall.HKEY_LOCAL_MACHINE, path, 0, syscall.KEY_READ, &k) != nil {
		return ""
	}
	defer syscall.RegCloseKey(k)
	name, _ := syscall.UTF16PtrFromString("ProcessorNameString")
	var buf [256]uint16
	n, typ := uint32(len(buf)*2), uint32(0)
	if syscall.RegQueryValueEx(k, name, nil, &typ, (*byte)(unsafe.Pointer(&buf[0])), &n) != nil || typ != syscall.REG_SZ {
		return ""
	}
	return strings.Join(strings.Fields(syscall.UTF16ToString(buf[:])), " ")
}
