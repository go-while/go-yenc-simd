// Command yencbench prints how fast go-yenc-simd decodes on this machine, for
// every kernel the CPU can run:
//
//	GOEXPERIMENT=simd go run github.com/go-while/go-yenc-simd/cmd/yencbench@latest
//
// Without GOEXPERIMENT=simd it measures the plain Go decoder.
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/go-while/go-yenc-simd/yencbench"
)

func main() {
	t := flag.Duration("time", time.Second, "time per measurement")
	size := flag.Int("size", 750_000, "data bytes per article")
	k := flag.String("kernels", "", "comma-separated kernels to measure, e.g. avx2,scalar (default: all)")
	flag.Parse()
	var kernels []string
	if *k != "" {
		kernels = strings.Split(*k, ",")
	}
	if err := yencbench.Run(os.Stdout, yencbench.Options{Time: *t, Size: *size, Kernels: kernels}); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
