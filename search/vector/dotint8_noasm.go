//go:build !amd64

package vector

// dotInt8 falls back to the portable loop on architectures without the AVX2
// kernel.
func dotInt8(a, b []byte) int32 { return dotInt8Generic(a, b) }
