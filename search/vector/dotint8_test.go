package vector

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// refDotInt8 is an independent reference (no shared code with dotInt8Generic).
func refDotInt8(a, b []byte) int32 {
	var s int64
	for i := range a {
		s += int64(int8(a[i])) * int64(int8(b[i]))
	}
	return int32(s)
}

// TestDotInt8MatchesReference is the correctness gate: the dispatched dotInt8
// (the AVX2 kernel on an AVX2 machine) must equal the independent reference for
// every length — including non-multiples of 16 (tail), tiny sizes, and the
// signed extremes.
func TestDotInt8MatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	dims := []int{0, 1, 2, 7, 15, 16, 17, 31, 32, 33, 63, 64, 100, 127, 128, 255, 256, 1023, 1024}
	for _, n := range dims {
		for iter := 0; iter < 50; iter++ {
			a := make([]byte, n)
			b := make([]byte, n)
			rng.Read(a)
			rng.Read(b)
			got := dotInt8(a, b)
			gen := dotInt8Generic(a, b)
			ref := refDotInt8(a, b)
			if got != ref || gen != ref {
				t.Fatalf("n=%d: dotInt8=%d generic=%d ref=%d", n, got, gen, ref)
			}
		}
	}
}

// TestDotInt8Extremes checks the signed edges: all +127, all -128, mixed.
func TestDotInt8Extremes(t *testing.T) {
	for _, n := range []int{16, 17, 64, 1024} {
		cases := [][2]byte{{127, 127}, {0x80, 0x80}, {127, 0x80}, {0x80, 127}}
		for _, c := range cases {
			a := make([]byte, n)
			b := make([]byte, n)
			for i := range a {
				a[i], b[i] = c[0], c[1]
			}
			if got, ref := dotInt8(a, b), refDotInt8(a, b); got != ref {
				t.Fatalf("n=%d a=%d b=%d: got %d != ref %d", n, int8(c[0]), int8(c[1]), got, ref)
			}
		}
	}
}

// TestScalar8SymUnaffected: SimCodes still produces the same values as a direct
// scalar computation (the SIMD path must not change results, only speed).
func TestScalar8SymUnaffected(t *testing.T) {
	dim := 1024
	q := NewScalar8Sym(dim)
	vecs := randVecs(40, dim, 6, 3)
	codes := make([][]byte, len(vecs))
	for i, v := range vecs {
		codes[i] = q.Encode(v)
	}
	for i := range codes {
		for j := range codes {
			ci, cj := codes[i], codes[j]
			var d int32
			for x := 0; x < dim; x++ {
				d += int32(int8(ci[4+x])) * int32(int8(cj[4+x]))
			}
			sa := math.Float32frombits(binary.LittleEndian.Uint32(ci))
			sb := math.Float32frombits(binary.LittleEndian.Uint32(cj))
			want := sa * sb * float32(d)
			if got := q.SimCodes(ci, cj); got != want {
				t.Fatalf("SimCodes(%d,%d)=%v != scalar recompute %v", i, j, got, want)
			}
		}
	}
}

func BenchmarkDotInt8_1024(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	x := make([]byte, 1024)
	y := make([]byte, 1024)
	rng.Read(x)
	rng.Read(y)
	b.Run("generic", func(b *testing.B) {
		var s int32
		for i := 0; i < b.N; i++ {
			s += dotInt8Generic(x, y)
		}
		_ = s
	})
	b.Run("dispatched", func(b *testing.B) {
		var s int32
		for i := 0; i < b.N; i++ {
			s += dotInt8(x, y)
		}
		_ = s
	})
}
