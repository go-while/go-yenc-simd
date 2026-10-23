// Package e2e tests go-yenc-simd end to end. Posts come from nntp-testkit's fake NNTP server.
// They are fetched over TCP with BODY and decoded by Reader, the way the README shows:
//   - every data profile of nntp-testkit/yenc
//   - both encoder styles for the line-edge characters, line lengths 64 to 997, and an encoder
//     that escapes more than it must
//   - sizes from 0 bytes to 750 KB, and multi-part posts
//   - broken posts, after which the connection must still work
//   - posts cut off mid-body, posts written a byte or a line at a time, and parallel connections
//
// Every test runs for each kernel the CPU has. A last test decodes the encoder's posts with
// rapidyenc (its C library with cgo), so the encoder and Reader cannot share a mistake unseen.
//
// It is a module of its own so that the library keeps no dependencies:
//
//	cd e2e && go test ./... && GOEXPERIMENT=simd go test ./...
package e2e
