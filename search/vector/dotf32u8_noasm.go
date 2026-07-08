//go:build !amd64

package vector

// dotF32U8 falls back to the portable loop on architectures without the kernel.
func dotF32U8(q []float32, b []byte) float32 { return dotF32U8Generic(q, b) }
