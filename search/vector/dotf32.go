package vector

// dotFloat32Generic is the portable float32 dot product: a strict left-to-right
// sum. It is the fallback where the AVX2 kernel is unavailable, and computes the
// tail the kernel leaves.
func dotFloat32Generic(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
