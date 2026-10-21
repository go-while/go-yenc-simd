//go:build goexperiment.simd && amd64

package yencsimd

import (
	"bytes"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"testing"
)

// TestKernelISA reads the machine code of the test binary and checks that
// each kernel uses only instructions its gate guarantees, and that the code
// every amd64 CPU runs (package init, the API, the scalar decoder, the
// Reader) uses nothing beyond baseline x86-64: one instruction too many is a
// SIGILL on some CPU. steps16, the kernel of CPUs with AVX but not AVX2,
// once built its constants with VPBROADCASTB, an AVX2 instruction
// (archsimd.BroadcastUint8x16 says "Emulated, CPU Feature: AVX2").
//
// It disassembles the package (and the simd/archsimd functions it calls)
// with go tool objdump and classifies each instruction by its encoding (VEX,
// EVEX), its registers (Y, K) and its mnemonic (isaMnemonics). It can't see
// the ifs inside a function, so a kernel is a set of whole functions
// (isaSets). What GOAMD64 guarantees is allowed everywhere: a v3 binary
// doesn't start on a CPU without AVX2. The binary go test runs has no symbol
// table, so the test builds one that has (go test -c) unless it runs one.
func TestKernelISA(t *testing.T) {
	gocmd, err := exec.LookPath("go") // go test puts its GOROOT/bin first in PATH
	if err != nil {
		t.Skip("no go command for go tool objdump:", err)
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	level, levelHas := isaLevel()
	if levelHas != 0 {
		t.Logf("GOAMD64=%s: the binary needs %v on its own, so every function may use it", level, levelHas)
	}
	pkg := reflect.TypeFor[State]().PkgPath() + "."
	re := `^(` + regexp.QuoteMeta(pkg) + `|simd/archsimd\.)`
	funcs, err := isaDisassemble(t, gocmd, exe, re)
	if err != nil {
		// go test links the binary it runs without a symbol table (-s -w)
		// unless it keeps it (-c, -o, profiles): build one that has it
		t.Logf("%v; building the test binary again, with symbols", err)
		exe = isaBuild(t, gocmd)
		if funcs, err = isaDisassemble(t, gocmd, exe, re); err != nil {
			t.Fatal(err)
		}
	}

	name := func(f any) string { // without pkg
		return strings.TrimPrefix(runtime.FuncForPC(reflect.ValueOf(f).Pointer()).Name(), pkg)
	}
	listed := map[string]int{} // the set of each function in isaSets
	for i, s := range isaSets {
		for _, f := range s.funcs {
			n := name(f)
			if funcs[pkg+n] == nil {
				t.Errorf("%s (%s) is not in the disassembly of %s", n, s.kernel, exe)
			}
			listed[n] = i
		}
	}
	setOf := func(n string) int { // n's set, or its function's if it is a closure (steps16.func1)
		for {
			if i, ok := listed[n]; ok {
				return i
			}
			j := strings.LastIndexByte(n, '.')
			if j < 0 {
				return 0 // every CPU
			}
			n = n[:j]
		}
	}
	guarded := map[[2]string]bool{} // caller, callee
	for _, s := range isaSets {
		for _, c := range s.calls {
			guarded[[2]string{name(c[0]), name(c[1])}] = true
		}
	}

	type fnSet struct {
		fn  string // with its package
		set int
	}
	type job struct {
		fnSet
		from string // the caller, for a function of simd/archsimd
	}
	var todo []job
	for fn, f := range funcs {
		if n, ok := strings.CutPrefix(fn, pkg); ok && !strings.HasSuffix(f.file, "_test.go") {
			todo = append(todo, job{fnSet{fn, setOf(n)}, ""})
		}
	}
	done := map[fnSet]bool{}
	uses := make([]isaFeature, len(isaSets)) // what each set's code uses: proof that the disassembly was read
	for len(todo) > 0 {
		j := todo[len(todo)-1]
		todo = todo[:len(todo)-1]
		if done[j.fnSet] {
			continue
		}
		done[j.fnSet] = true
		s := isaSets[j.set]
		has := s.has | levelHas
		what := "code every CPU runs; if a feature check guards it, list it in isaSets"
		if j.set != 0 {
			what = "kernel " + s.kernel + ", gate " + s.gate
		}
		if j.from != "" {
			what = "called by " + j.from + ", " + what
		}
		caller := strings.TrimPrefix(j.fn, pkg)
		type bad struct {
			n     int
			first isaInst
		}
		bads := map[string]*bad{} // by mnemonic and missing feature
		var order []string
		for _, in := range funcs[j.fn].insts {
			need := isaNeeds(in)
			uses[j.set] |= need
			if miss := need &^ has; miss != 0 {
				op, _, _ := strings.Cut(in.text, " ")
				key := op + " needs " + miss.String()
				if bads[key] == nil {
					bads[key] = &bad{first: in}
					order = append(order, key)
				}
				bads[key].n++
			}
			callee := isaCallee(in.text)
			if strings.HasPrefix(callee, "simd/archsimd.") && funcs[callee] != nil {
				todo = append(todo, job{fnSet{callee, j.set}, caller}) // runs with the caller's features
			} else if n, ok := strings.CutPrefix(callee, pkg); ok {
				if cs := setOf(n); isaSets[cs].has&^s.has != 0 && !guarded[[2]string{caller, n}] {
					t.Errorf("%s (%s) calls %s (kernel %s) at %s %#x: if a feature check guards the call, list it in isaSets",
						caller, what, n, isaSets[cs].kernel, in.pos, in.addr)
				}
			}
		}
		for _, key := range order {
			b := bads[key]
			t.Errorf("%s (%s): %s, %d times, first at %s %#x: %s",
				caller, what, key, b.n, b.first.pos, b.first.addr, b.first.text)
		}
	}
	for i, s := range isaSets {
		if s.has != 0 && uses[i]&s.has == 0 {
			t.Errorf("kernel %s: no vector instructions found in it; is the disassembly read right?", s.kernel)
		}
	}
}

// isaSets are the kernels, the features their gate guarantees and the
// functions of the package that only they run (with their closures). Every
// other function of the package runs on every CPU: one with vector code
// must be listed here, under the weakest kernel that runs it. calls are the
// calls of a function of the set to one of a kernel with more features,
// behind its gate.
var isaSets = []struct {
	kernel string
	gate   string
	has    isaFeature
	funcs  []any
	calls  [][2]any
}{
	{"every CPU", "", 0,
		// listed so that they are in the test binary (no test calls
		// Kernels or SetKernel) and the check sees them
		[]any{Kernel, Kernels, SetKernel, DecodeIncremental, Decode, NewReader},
		[][2]any{{decode, decodeVBMI2}, {decode, decodeSIMD}}}, // behind hasVBMI2, hasAVX
	// hasVBMI2 is AVX512 (F+CD+BW+DQ+VL, which includes AVX and AVX2) and
	// VBMI (VPERMI2B), VBMI2 (VPCOMPRESSB) and VPOPCNTDQ (VPOPCNTQ; on X
	// registers it needs VL as well, which AVX512 has)
	{"avx512vbmi2", "hasVBMI2 (AVX512 F+CD+BW+DQ+VL, VBMI, VBMI2, VPOPCNTDQ)",
		fAVX | fYMM | fAVX2 | fAVX512 | fVBMI | fVBMI2 | fVPOPCNTDQ,
		[]any{decodeVBMI2, tailVBMI2}, nil},
	// hasAVX2 is AVX2 and AVX; no AVX-512 at all (no EVEX, no X16-X31, no
	// mask registers), no FMA, no BMI
	{"avx2", "hasAVX2 (AVX2)", fAVX | fYMM | fAVX2, []any{steps32}, nil},
	// hasAVX is AVX only: Sandy and Ivy Bridge, AMD Bulldozer and Jaguar,
	// VMs that hide AVX2. VEX.128 instructions, no Y registers, none of the
	// AVX2 instructions that have 128-bit forms (VPBROADCASTB X, X).
	{"avx", "hasAVX (AVX)", fAVX, []any{decodeSIMD, steps16},
		[][2]any{{decodeSIMD, steps32}}}, // behind hasAVX2
}

// isaFeature is a set of what instructions need beyond baseline x86-64.
type isaFeature uint16

const (
	fAVX       isaFeature = 1 << iota // VEX encoding: VEX.128, VZEROUPPER
	fYMM                              // Y registers (VEX.256): AVX, AVX2 for integers
	fAVX2                             // AVX2 only, at any width
	fBMI                              // BMI1, BMI2, LZCNT, MOVBE
	fFMA                              // FMA3
	fF16C                             // F16C
	fAVX512                           // EVEX or mask registers: AVX512 F+CD+BW+DQ+VL
	fVBMI                             // AVX512VBMI
	fVBMI2                            // AVX512VBMI2
	fVPOPCNTDQ                        // AVX512VPOPCNTDQ
	fOther                            // an extension no gate checks
	fUnknown                          // objdump can't decode it
)

var isaFeatureNames = [...]string{"AVX", "Y registers", "AVX2", "BMI/LZCNT/MOVBE", "FMA", "F16C",
	"AVX-512", "AVX512VBMI", "AVX512VBMI2", "AVX512VPOPCNTDQ", "an extension no gate checks", "a check by hand (objdump can't decode it)"}

func (f isaFeature) String() string {
	var s []string
	for i, n := range isaFeatureNames {
		if f&(1<<i) != 0 {
			s = append(s, n)
		}
	}
	return strings.Join(s, "+")
}

// isaLevels are what each GOAMD64 level guarantees by itself: the runtime
// of a v3 binary doesn't start on a CPU without AVX2, BMI2, FMA, ..., and
// the compiler uses them everywhere (SHLX, TZCNT, ...). v2 adds SSE3 to
// SSE4.2 and POPCNT, which aren't VEX and aren't checked.
var isaLevels = map[string]isaFeature{
	"v1": 0,
	"v2": 0,
	"v3": fAVX | fYMM | fAVX2 | fBMI | fFMA | fF16C,
	"v4": fAVX | fYMM | fAVX2 | fBMI | fFMA | fF16C | fAVX512,
}

// isaLevel returns the GOAMD64 level the test binary was built for (from its
// build info, or the default v1) and what it guarantees.
func isaLevel() (string, isaFeature) {
	level := "v1"
	if bi, ok := debug.ReadBuildInfo(); ok {
		for _, s := range bi.Settings {
			if s.Key == "GOAMD64" {
				level = s.Value
			}
		}
	}
	return level, isaLevels[level]
}

// isaMnemonics are the instructions that need more than their encoding and
// registers show: AVX2 instructions with 128-bit forms, the VEX instructions
// on general registers and those of the extensions. Names are Go assembler
// syntax, as objdump prints them, without a suffix like .Z or .BCST.
var isaMnemonics = []struct {
	f  isaFeature
	re *regexp.Regexp
}{
	{fAVX2, regexp.MustCompile(`^(VPBROADCAST[BWDQ]|VBROADCASTI128|VPERM2I128|VINSERTI128|VEXTRACTI128|VPERM[DQ]|VPERMP[DS]|VPBLENDD|VPS[LR]LV[DQ]|VPSRAVD|VPMASKMOV[DQ]|VPGATHER[DQ][DQ]|VGATHER[DQ]P[DS])$`)},
	{fBMI, regexp.MustCompile(`^(ANDN|BEXTR|BLSI|BLSMSK|BLSR|BZHI|MULX|PDEP|PEXT|RORX|SARX|SHLX|SHRX|TZCNT|LZCNT|MOVBE)[WLQ]?$`)},
	{fFMA, regexp.MustCompile(`^VF(N?M(ADD|SUB)|MADDSUB|MSUBADD)`)},
	{fF16C, regexp.MustCompile(`^VCVT(PH2PS|PS2PH)$`)},
	{fVBMI, regexp.MustCompile(`^(VPERMB|VPERM[IT]2B|VPMULTISHIFTQB)$`)},
	{fVBMI2, regexp.MustCompile(`^(VPCOMPRESS[BW]|VPEXPAND[BW]|VPSH[LR]DV?[WDQ])$`)},
	{fVPOPCNTDQ, regexp.MustCompile(`^VPOPCNT[DQ]$`)},
	// AES, PCLMULQDQ, GFNI, SHA (archsimd has these), SM3, SM4, BITALG,
	// (AVX-)VNNI, IFMA, BF16, AVX-NE-CONVERT, VP2INTERSECT, Xeon Phi, FP16
	{fOther, regexp.MustCompile(`^(V?AES|V?PCLMUL|V?GF2P8|V?SHA|VSM[34]|VPOPCNT[BW]$|VPSHUFBITQMB$|VPDP[BW]|VPMADD52|VCVTNE|VBCSTNE|VDPBF16|VP2INTERSECT|V4F|VP4DP|VEXP2|VRCP28|VRSQRT28|V(GATHER|SCATTER)PF|V.*[^2][PS]H$|VCVT(U?W|U?DQ|U?QQ|PD|SS|SD|U?SI)2[PS]H$|VCVTP[HS]2P[HS]X$)`)},
}

var (
	isaRegY   = regexp.MustCompile(`\bY([0-9]|1[0-5])\b`) // VEX has 16 registers
	isaRegK   = regexp.MustCompile(`\bK[0-7]\b`)
	isaSymbol = regexp.MustCompile(`\S+\(SB\)`)
	isaCall   = regexp.MustCompile(`^(?:CALL|JMP) (\S+)\(SB\)$`)
)

// isaNeeds returns what in needs beyond baseline x86-64.
func isaNeeds(in isaInst) isaFeature {
	if in.text == "?" {
		return fUnknown
	}
	op, args, _ := strings.Cut(in.text, " ")
	op, _, _ = strings.Cut(op, ".")
	args = isaSymbol.ReplaceAllString(args, "")
	var f isaFeature
	for _, m := range isaMnemonics {
		if m.re.MatchString(op) {
			f |= m.f
		}
	}
	if (op == "VBROADCASTSS" || op == "VBROADCASTSD") && strings.HasPrefix(args, "X") {
		f |= fAVX2 // from a register; from memory it is AVX
	}
	code := bytes.TrimLeft(in.code, "\x26\x2e\x36\x3e\x64\x65\x67") // segment and address size prefixes
	switch {
	case len(code) == 0:
	case code[0] == 0x62: // EVEX (BOUND in 32-bit mode only)
		f |= fAVX512
	case code[0] == 0xc4 || code[0] == 0xc5: // VEX (LES, LDS in 32-bit mode only)
		switch {
		case strings.HasPrefix(op, "K") || isaRegK.MatchString(args):
			f |= fAVX512 // KMOVQ and the other mask instructions
		case f&fBMI == 0: // not ANDN, SHLX, ... on general registers
			f |= fAVX
			if isaRegY.MatchString(args) {
				f |= fYMM
			}
		}
	}
	return f
}

// isaCallee returns the function a CALL or JMP instruction goes to, if it
// names one.
func isaCallee(text string) string {
	if m := isaCall.FindStringSubmatch(text); m != nil {
		return m[1]
	}
	return ""
}

// isaInst is an instruction as go tool objdump prints it.
type isaInst struct {
	pos  string // file:line
	addr uint64
	code []byte
	text string // Go assembler syntax
}

type isaFunc struct {
	file  string // its source file, <autogenerated> for init
	insts []isaInst
}

// isaBuild builds the test binary again with go test -c, which keeps the
// symbol table, and returns its path. The settings that change the code come
// from the build info of the running one.
func isaBuild(t *testing.T, gocmd string) string {
	t.Helper()
	exe := filepath.Join(t.TempDir(), "isa.test")
	args := []string{"test", "-c", "-o", exe}
	env := os.Environ()
	if bi, ok := debug.ReadBuildInfo(); ok {
		for _, s := range bi.Settings {
			switch s.Key {
			case "-asan", "-cover", "-gcflags", "-msan", "-race", "-tags", "-trimpath":
				args = append(args, s.Key+"="+s.Value)
			case "CGO_ENABLED", "GOAMD64", "GOEXPERIMENT":
				env = append(env, s.Key+"="+s.Value)
			}
		}
	}
	cmd := exec.Command(gocmd, append(args, ".")...)
	cmd.Env = env
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("%v: %v\n%s", cmd.Args, err, out)
	}
	return exe
}

// isaDisassemble returns the functions of exe whose names match re.
//
// go1.27's disassembler (x86asm) misreads two kinds of instruction, and may
// decode what follows out of step (it showed PUSHQ SP after the VZEROUPPER
// in decodeSIMD): it reads a ModRM byte after VZEROUPPER and VZEROALL, which
// have none, and it can't decode the VEX instructions on general registers
// (BMI: ANDN, SHLX, ...), which GOAMD64=v3 code has: "?". They are replaced
// with NOPs in a copy of exe, which is disassembled again until objdump
// reads every function of the package, and put back in the listing.
func isaDisassemble(t *testing.T, gocmd, exe, re string) (map[string]*isaFunc, error) {
	t.Helper()
	funcs, err := isaObjdump(gocmd, exe, re)
	if err != nil {
		return nil, err
	}
	var data []byte               // exe, then the copy with NOPs
	fixed := map[uint64]isaInst{} // the instructions replaced, by address
	for round := 0; ; round++ {
		n := 0
		for fn, f := range funcs {
			if strings.HasSuffix(f.file, "_test.go") {
				continue // not checked
			}
			var body []byte             // the function's code, to find it in the file
			starts := map[uint64]bool{} // where objdump starts an instruction
			for _, in := range f.insts {
				body = append(body, in.code...)
				starts[in.addr] = true
			}
			var fix []isaInst
			for _, in := range f.insts {
				if v := isaVzeroLen(in); in.text != "?" && (v == 0 || len(in.code) == v) {
					continue // read right
				}
				r := isaReread(in, body[in.addr-f.insts[0].addr:])
				if r.code == nil {
					break // something else: TestKernelISA fails on it
				}
				fix = append(fix, r)
				if !starts[r.addr+uint64(len(r.code))] {
					break // objdump goes on out of step: the next round
				}
			}
			if len(fix) == 0 {
				continue
			}
			if data == nil {
				if data, err = os.ReadFile(exe); err != nil {
					t.Fatal(err)
				}
			}
			off := bytes.Index(data, body)
			if off < 0 || bytes.Contains(data[off+1:], body) {
				t.Fatalf("can't find %s in %s once, to disassemble it again", fn, exe)
			}
			for _, in := range fix {
				p := off + int(in.addr-f.insts[0].addr)
				copy(data[p:p+len(in.code)], bytes.Repeat([]byte{0x90}, len(in.code)))
				fixed[in.addr] = in
			}
			n += len(fix)
		}
		if n == 0 {
			break
		}
		if round == 20 {
			t.Fatal("objdump keeps misreading instructions")
		}
		patched := filepath.Join(t.TempDir(), filepath.Base(exe))
		if err := os.WriteFile(patched, data, 0o644); err != nil {
			t.Fatal(err)
		}
		if funcs, err = isaObjdump(gocmd, patched, re); err != nil {
			t.Fatal(err)
		}
	}
	if len(fixed) > 0 {
		t.Logf("objdump misread %d instructions (VZEROUPPER, BMI); read them again", len(fixed))
	}
	for _, f := range funcs {
		insts := f.insts[:0]
		for i := 0; i < len(f.insts); i++ {
			if in, ok := fixed[f.insts[i].addr]; ok {
				insts = append(insts, in)
				i += len(in.code) - 1 // the other NOPs
				continue
			}
			insts = append(insts, f.insts[i])
		}
		f.insts = insts
	}
	return funcs, nil
}

// isaReread returns the instruction objdump printed as in, read from code
// (its bytes and those after it), if in is a misread VZEROUPPER or VZEROALL
// or a BMI instruction objdump can't decode; else its code is nil.
func isaReread(in isaInst, code []byte) isaInst {
	if n := isaVzeroLen(in); n > 0 && len(in.code) > n {
		return isaInst{in.pos, in.addr, code[:n:n], in.text}
	}
	if name, n := isaBMI(code); n > 0 {
		return isaInst{in.pos, in.addr, code[:n:n], name}
	}
	return isaInst{}
}

// isaVzeroLen returns the length of in if it is VZEROUPPER or VZEROALL: a
// VEX prefix of 2 or 3 bytes and the opcode, no ModRM. 0 for others.
func isaVzeroLen(in isaInst) int {
	switch {
	case in.text != "VZEROUPPER" && in.text != "VZEROALL":
		return 0
	case in.code[0] == 0xc5:
		return 3
	}
	return 4
}

// isaBMI decodes the BMI1 and BMI2 instructions on general registers, which
// have a 3-byte VEX prefix (map 0F38, or 0F3A for RORX), and returns the name
// and length of the one code starts with, or 0.
func isaBMI(code []byte) (string, int) {
	if len(code) < 5 || code[0] != 0xc4 {
		return "", 0
	}
	m, w, pp, op, modrm := code[1]&0x1f, code[2]>>7, code[2]&3, code[3], code[4]
	var name string
	switch {
	case m == 2 && op == 0xf2 && pp == 0:
		name = "ANDN"
	case m == 2 && op == 0xf3 && pp == 0:
		name = [8]string{1: "BLSR", 2: "BLSMSK", 3: "BLSI"}[modrm>>3&7]
	case m == 2 && op == 0xf5: // pp: none, 66, F3, F2
		name = [4]string{"BZHI", "", "PEXT", "PDEP"}[pp]
	case m == 2 && op == 0xf6 && pp == 3:
		name = "MULX"
	case m == 2 && op == 0xf7:
		name = [4]string{"BEXTR", "SHLX", "SARX", "SHRX"}[pp]
	case m == 3 && op == 0xf0 && pp == 3:
		name = "RORX"
	}
	if name == "" {
		return "", 0
	}
	n := 5 // C4, 2 bytes, the opcode and ModRM
	if mod, rm := modrm>>6, modrm&7; mod != 3 {
		if rm == 4 { // SIB
			if len(code) > 5 && mod == 0 && code[5]&7 == 5 {
				n += 4 // no base: disp32
			}
			n++
		}
		switch {
		case mod == 1:
			n++
		case mod == 2, mod == 0 && rm == 5:
			n += 4
		}
	}
	if m == 3 {
		n++ // RORX's imm8
	}
	if n > len(code) {
		return "", 0
	}
	return name + [2]string{"L", "Q"}[w], n
}

// isaObjdump runs go tool objdump on the functions of exe whose names match
// re and returns them by name.
func isaObjdump(gocmd, exe, re string) (map[string]*isaFunc, error) {
	out, err := exec.Command(gocmd, "tool", "objdump", "-s", re, exe).Output()
	if err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			err = fmt.Errorf("%v: %s", err, bytes.TrimSpace(ee.Stderr))
		}
		return nil, fmt.Errorf("go tool objdump %s: %v", exe, err)
	}
	funcs := map[string]*isaFunc{}
	var f *isaFunc
	for line := range strings.Lines(string(out)) {
		// TEXT pkg.name(SB) file
		if rest, ok := strings.CutPrefix(line, "TEXT "); ok {
			fn, file, _ := strings.Cut(strings.TrimSpace(rest), "(SB) ")
			f = &isaFunc{file: file}
			funcs[fn] = f
			continue
		}
		// "  file.go:12\t0x4a1b2c\t\tc5f877\t\tVZEROUPPER\t...", padded with tabs
		var fields []string
		for s := range strings.SplitSeq(line, "\t") {
			if s = strings.TrimSpace(s); s != "" {
				fields = append(fields, s)
			}
		}
		if f == nil || len(fields) < 4 {
			continue
		}
		addr, err1 := strconv.ParseUint(strings.TrimPrefix(fields[1], "0x"), 16, 64)
		code, err2 := hex.DecodeString(fields[2])
		if err1 != nil || err2 != nil || len(code) == 0 {
			return nil, fmt.Errorf("go tool objdump %s: can't read %q", exe, line)
		}
		f.insts = append(f.insts, isaInst{fields[0], addr, code, fields[3]})
	}
	if len(funcs) == 0 { // a PE file without symbols
		return nil, fmt.Errorf("go tool objdump %s: no functions match %s", exe, re)
	}
	return funcs, nil
}
