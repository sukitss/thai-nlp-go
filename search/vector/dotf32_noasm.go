//go:build !amd64

package vector

// dot falls back to the portable loop on architectures without the AVX2 kernel.
func dot(a, b []float32) float32 { return dotFloat32Generic(a, b) }
