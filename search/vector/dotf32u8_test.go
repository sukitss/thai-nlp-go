package vector

import (
	"math"
	"math/rand"
	"testing"
)

// TestDotF32U8MatchesGeneric: the dispatched dotF32U8 (AVX2 on an AVX2 machine)
// agrees with the strict sequential float32-times-unsigned-byte sum within
// float32 rounding, across lengths incl. tails, tiny, and 1024.
func TestDotF32U8MatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	dims := []int{0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 64, 100, 1023, 1024}
	for _, n := range dims {
		for iter := 0; iter < 50; iter++ {
			q := make([]float32, n)
			b := make([]byte, n)
			for i := range q {
				q[i] = float32(rng.NormFloat64())
			}
			rng.Read(b)
			got := dotF32U8(q, b)
			ref := dotF32U8Generic(q, b)
			tol := 1e-4 * (float32(math.Abs(float64(ref))) + 1)
			if d := got - ref; d > tol || d < -tol {
				t.Fatalf("n=%d: dotF32U8=%v generic=%v diff=%v tol=%v", n, got, ref, d, tol)
			}
		}
	}
}
