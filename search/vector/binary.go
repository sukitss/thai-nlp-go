package vector

import (
	"encoding/binary"
	"math/bits"
)

// Binary quantizes each vector to one sign bit per dimension: bit i is 1 when
// vec[i] > 0. Similarity is the Hamming distance between bitsets, computed with
// hardware popcount ([bits.OnesCount64]). At one bit per dimension it is ~32x
// smaller than [Float32].
//
// For sign / random-hyperplane quantization the expected cosine relates to the
// Hamming distance h over d dimensions as cos(theta) ~= cos(pi*h/d); the linear
// proxy 1 - 2h/d used by Score is monotonic in h, so it ranks identically while
// staying a readable cosine-like number in [-1, 1].
type Binary struct {
	dim   int
	words int // uint64 words per code
}

// NewBinary returns a Binary quantizer for dim-dimensional vectors.
func NewBinary(dim int) *Binary {
	if dim <= 0 {
		panic("vector: NewBinary dim must be > 0")
	}
	return &Binary{dim: dim, words: (dim + 63) / 64}
}

func (q *Binary) Dim() int     { return q.dim }
func (q *Binary) CodeLen() int { return q.words * 8 }
func (q *Binary) Name() string { return "binary" }

// Encode packs the sign bits of vec little-endian, low bit first; trailing pad
// bits (dim not a multiple of 64) are zero.
func (q *Binary) Encode(vec []float32) []byte {
	if len(vec) != q.dim {
		panic("vector: Binary.Encode dim mismatch")
	}
	code := make([]byte, q.words*8)
	for i, x := range vec {
		if x > 0 {
			code[i>>3] |= 1 << (uint(i) & 7)
		}
	}
	return code
}

// Hamming returns the exact number of differing bits between two codes. Both
// must be CodeLen() bytes. It is the raw symmetric primitive behind Score.
func (q *Binary) Hamming(a, b []byte) int {
	var h int
	for w := 0; w < q.words; w++ {
		ua := binary.LittleEndian.Uint64(a[w*8:])
		ub := binary.LittleEndian.Uint64(b[w*8:])
		h += bits.OnesCount64(ua ^ ub)
	}
	return h
}

// Query encodes the query to a bitset once; Score returns 1 - 2h/dim.
func (q *Binary) Query(query []float32) Scorer {
	if len(query) != q.dim {
		panic("vector: Binary.Query dim mismatch")
	}
	return &binaryScorer{q: q, code: q.Encode(query)}
}

type binaryScorer struct {
	q    *Binary
	code []byte
}

func (s *binaryScorer) Score(code []byte) float32 {
	h := s.q.Hamming(s.code, code)
	return 1 - 2*float32(h)/float32(s.q.dim)
}
