package vector

import (
	"math"
	"math/rand"
	"testing"
)

// TestDotFloat32MatchesGeneric: the dispatched dot (AVX2 on an AVX2 machine)
// must agree with the strict sequential sum within float32 rounding — the AVX2
// path sums in parallel lanes, so it is close but not bit-identical. Checked over
// many lengths including non-multiples of 8 (tail), tiny sizes, and dim 1024.
func TestDotFloat32MatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	dims := []int{0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 64, 100, 1023, 1024}
	for _, n := range dims {
		for iter := 0; iter < 50; iter++ {
			a := make([]float32, n)
			b := make([]float32, n)
			for i := range a {
				a[i] = float32(rng.NormFloat64())
				b[i] = float32(rng.NormFloat64())
			}
			got := dot(a, b)
			ref := dotFloat32Generic(a, b)
			// relative tolerance: parallel-lane summation vs sequential.
			tol := 1e-4 * (float32(math.Abs(float64(ref))) + 1)
			if d := got - ref; d > tol || d < -tol {
				t.Fatalf("n=%d: dot=%v generic=%v diff=%v tol=%v", n, got, ref, d, tol)
			}
		}
	}
}

func BenchmarkDotFloat32_1024(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	x := make([]float32, 1024)
	y := make([]float32, 1024)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
		y[i] = float32(rng.NormFloat64())
	}
	b.Run("generic", func(b *testing.B) {
		var s float32
		for i := 0; i < b.N; i++ {
			s += dotFloat32Generic(x, y)
		}
		_ = s
	})
	b.Run("dispatched", func(b *testing.B) {
		var s float32
		for i := 0; i < b.N; i++ {
			s += dot(x, y)
		}
		_ = s
	})
}
