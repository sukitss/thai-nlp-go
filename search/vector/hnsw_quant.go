package vector

import (
	"encoding/binary"
	"math"
	"math/bits"
)

// SymQuantizer compresses float32 vectors into fixed-length byte codes AND
// scores two codes against each other with a SYMMETRIC similarity. This is what
// [WithQuantizer] needs and the query-bound [Quantizer]/[Scorer] cannot provide:
// building an HNSW graph links stored vector to stored vector, so it needs a
// code-to-code distance, not just query-to-code. Both the graph and the query
// run in this code space (the query is Encoded once, then compared to node
// codes), so it is a symmetric-distance (SDC) index — lower precision than the
// asymmetric [Quantizer.Query] path, in exchange for a graph that is 4–32×
// smaller and cheaper to build and search.
//
// SimCodes must be symmetric (SimCodes(a,b) == SimCodes(b,a)) and order
// similarity so that LARGER means MORE similar, matching the rest of the package.
type SymQuantizer interface {
	// Encode compresses vec (length Dim, expected L2-normalized by the caller)
	// into a code of length CodeLen.
	Encode(vec []float32) []byte
	// Dim is the vector dimensionality.
	Dim() int
	// CodeLen is the byte length of every code Encode produces.
	CodeLen() int
	// SimCodes reports the similarity of two codes (larger = more similar).
	SimCodes(a, b []byte) float32
	// Name identifies the quantizer for reporting.
	Name() string
}

// BinarySym is a 1-bit-per-dimension [SymQuantizer]: each dimension becomes its
// sign bit and similarity is the number of agreeing bits, dim − 2·Hamming,
// computed with hardware popcount. ~32× smaller than float32. Its similarity is
// a monotone function of the cosine of the sign vectors, so its ranking tracks
// cosine while its magnitude does not equal it.
type BinarySym struct{ dim int }

// NewBinarySym returns a binary sign-bit symmetric quantizer for dim dimensions.
func NewBinarySym(dim int) *BinarySym { return &BinarySym{dim: dim} }

func (b *BinarySym) Dim() int     { return b.dim }
func (b *BinarySym) CodeLen() int { return ((b.dim + 63) / 64) * 8 }
func (b *BinarySym) Name() string { return "binary-sym" }

func (b *BinarySym) Encode(vec []float32) []byte {
	code := make([]byte, b.CodeLen())
	for i := 0; i < b.dim; i++ {
		if vec[i] >= 0 { // sign bit set for non-negative components
			code[i>>3] |= 1 << (uint(i) & 7)
		}
	}
	return code
}

func (b *BinarySym) SimCodes(a, c []byte) float32 {
	var ham int
	i, n := 0, len(a)
	for ; i+8 <= n; i += 8 { // 64 bits at a time
		ham += bits.OnesCount64(binary.LittleEndian.Uint64(a[i:]) ^ binary.LittleEndian.Uint64(c[i:]))
	}
	for ; i < n; i++ { // tail (CodeLen is a multiple of 8, so normally unused)
		ham += bits.OnesCount8(a[i] ^ c[i])
	}
	// Padding bits beyond dim are 0 in every code, so they never differ and do
	// not inflate the Hamming distance.
	return float32(b.dim - 2*ham)
}

// Scalar8Sym is an int8-per-dimension [SymQuantizer] with a per-vector scale:
// each vector is quantized to int8 by its own max-abs scale and similarity is the
// scaled integer dot product. ~4× smaller than float32 and markedly more accurate
// than [BinarySym]. A code is a little-endian float32 scale followed by dim int8s.
type Scalar8Sym struct{ dim int }

// NewScalar8Sym returns an int8 symmetric quantizer for dim dimensions.
func NewScalar8Sym(dim int) *Scalar8Sym { return &Scalar8Sym{dim: dim} }

func (s *Scalar8Sym) Dim() int     { return s.dim }
func (s *Scalar8Sym) CodeLen() int { return 4 + s.dim }
func (s *Scalar8Sym) Name() string { return "scalar8-sym" }

func (s *Scalar8Sym) Encode(vec []float32) []byte {
	var maxAbs float32
	for _, x := range vec {
		if a := abs32(x); a > maxAbs {
			maxAbs = a
		}
	}
	scale := maxAbs / 127
	if scale == 0 {
		scale = 1 // all-zero vector: any scale works, codes are all zero
	}
	code := make([]byte, 4+s.dim)
	binary.LittleEndian.PutUint32(code, math.Float32bits(scale))
	for i, x := range vec {
		q := int32(math.Round(float64(x / scale)))
		if q > 127 {
			q = 127
		} else if q < -127 { // avoid -128 to keep the range symmetric
			q = -127
		}
		code[4+i] = byte(int8(q))
	}
	return code
}

func (s *Scalar8Sym) SimCodes(a, b []byte) float32 {
	sa := math.Float32frombits(binary.LittleEndian.Uint32(a))
	sb := math.Float32frombits(binary.LittleEndian.Uint32(b))
	var d int32
	for i := 0; i < s.dim; i++ {
		d += int32(int8(a[4+i])) * int32(int8(b[4+i]))
	}
	return sa * sb * float32(d)
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
