//go:build amd64

package vector

import "unsafe"

// dotF32Bytes dots q against a code holding little-endian float32s (as produced
// by Float32.Encode). On amd64 the machine is little-endian, so the code bytes
// are exactly the float32 values: it reinterprets them without copying and reuses
// the AVX2 dot kernel. len(code) must be >= 4*len(q).
func dotF32Bytes(q []float32, code []byte) float32 {
	cf := unsafe.Slice((*float32)(unsafe.Pointer(&code[0])), len(q))
	return dot(q, cf)
}
