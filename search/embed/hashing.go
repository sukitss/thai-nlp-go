package embed

import (
	"strings"
	"unicode/utf8"

	"github.com/sukitss/thai-nlp-go/multi"
	"github.com/sukitss/thai-nlp-go/vocab"
)

// Hashing is a feature-hashing (a.k.a. "hashing trick", Weinberger et al. 2009)
// embedder over character n-grams. It needs NO pretrained model: it tokenizes
// the text, expands every token into its character n-grams (plus, optionally,
// the whole token), hashes each feature to a dimension with a signed hash, sums
// the signed contributions, and L2-normalizes the result. The dimension is
// fixed and chosen by the caller; the hash makes the mapping deterministic and
// collision-controlled (a larger dim means fewer collisions).
//
// # Why character n-grams — the lexical-similarity property
//
// Two spellings of the same name share most of their character n-grams, so they
// hash to a heavily overlapping set of dimensions and end up with a HIGH cosine
// similarity even though no whole-token or exact term matches. This is the whole
// point: it catches variants that exact BM25 misses —
//
//	"ปลา IT"  → tokens {ปลา, it}   → grams {^ปล, ปลา, ลา$, ^it, it$, ...}
//	"ปลาit"   → tokens {ปลา, it}   → grams {^ปล, ปลา, ลา$, ^it, it$, ...}
//
// so a query "ปลาit หน่อย" lands close to a document about "ปลา IT" despite the
// stuck acronym and missing space. Tokens are lower-cased before feature
// extraction, so "IT" and "it" produce identical character features (the fuzzy
// path deliberately folds the case distinction the exact/keyword path keeps).
//
// It captures LEXICAL similarity only. Synonyms with no shared characters
// ("รถ" vs "ยานพาหนะ") get NO similarity from this embedder — for that you need
// distributional semantics ([Static] with a real model). Use Hashing for
// names, typos, transliterations, and acronym/spacing variants; use it ALONGSIDE
// BM25 (fuse with reciprocal-rank fusion), where it recovers the variant-spelling
// slice BM25 cannot.
//
// # Optional IDF weighting
//
// With [WithIDF] each token's features are weighted by the token's BM25 IDF from
// a supplied [vocab.Vocab], so common words ("การ", "ที่") contribute less than
// rare discriminative ones — the same rare-term emphasis BM25 gives sparse
// retrieval, carried into the dense vector.
type Hashing struct {
	dim      int
	minN     int
	maxN     int
	seed     uint64
	wordFeat bool
	charFeat bool
	analyzer *multi.Analyzer
	idf      *vocab.Vocab
}

// HashingOption configures a [Hashing] at construction.
type HashingOption func(*Hashing)

// WithNGrams sets the inclusive character n-gram range (rune-based). The default
// is 2..4, which balances discrimination (longer grams are more specific)
// against variant-robustness (shorter grams survive an edit). minN < 1 is raised
// to 1; maxN < minN is raised to minN.
func WithNGrams(minN, maxN int) HashingOption {
	return func(h *Hashing) { h.minN, h.maxN = minN, maxN }
}

// WithSeed sets the hash seed, which fixes the (feature → dimension, sign)
// mapping. Any two Hashing embedders with the same dim, n-gram range and seed
// produce identical vectors, so persisted vectors stay comparable across
// processes only if the seed matches. Default: a fixed non-zero seed.
func WithSeed(seed uint64) HashingOption {
	return func(h *Hashing) { h.seed = seed }
}

// WithFeatures selects which feature families are emitted: character n-grams
// (the variant-robust signal) and/or the whole lower-cased token (an exact-match
// signal in a separate hash namespace). At least one must be enabled; disabling
// both is corrected to char-only. Default: both on.
func WithFeatures(char, word bool) HashingOption {
	return func(h *Hashing) { h.charFeat, h.wordFeat = char, word }
}

// WithAnalyzer sets the tokenization front-end (script routing, overlays). The
// default is the zero [multi.Analyzer] (normalize + route + tokenize), which is
// enough because Hashing lower-cases tokens itself; set your own only to add a
// per-tenant dictionary overlay or CJK options.
func WithAnalyzer(a *multi.Analyzer) HashingOption {
	return func(h *Hashing) { h.analyzer = a }
}

// WithIDF weights each token's features by the token's BM25 IDF from v (see the
// type doc). nil disables weighting (every token weighs 1). The vocab should be
// built over the SAME analysis as the corpus for the weights to line up.
func WithIDF(v *vocab.Vocab) HashingOption {
	return func(h *Hashing) { h.idf = v }
}

// NewHashing returns a Hashing embedder producing dim-dimensional vectors. dim
// must be > 0 (a larger dim means fewer hash collisions and a more faithful
// embedding). See the With* options for n-gram range, seeding, feature families,
// analyzer and IDF weighting.
func NewHashing(dim int, opts ...HashingOption) *Hashing {
	if dim <= 0 {
		panic("embed: NewHashing dim must be > 0")
	}
	h := &Hashing{
		dim:      dim,
		minN:     2,
		maxN:     4,
		seed:     fnvOffset,
		wordFeat: true,
		charFeat: true,
	}
	for _, o := range opts {
		o(h)
	}
	if h.minN < 1 {
		h.minN = 1
	}
	if h.maxN < h.minN {
		h.maxN = h.minN
	}
	if !h.charFeat && !h.wordFeat {
		h.charFeat = true
	}
	if h.analyzer == nil {
		h.analyzer = &multi.Analyzer{}
	}
	return h
}

// Dim reports the embedding dimensionality.
func (h *Hashing) Dim() int { return h.dim }

// Embed tokenizes text and returns the L2-normalized feature-hashing vector (see
// the type doc). Empty or content-free text yields an all-zero vector.
func (h *Hashing) Embed(text string) []float32 {
	acc := make([]float32, h.dim)
	toks := h.analyzer.Terms(text)
	for _, tok := range toks {
		tok = strings.ToLower(tok) // case-fold: "IT"/"it" share features
		w := float32(1)
		if h.idf != nil {
			w = float32(h.idf.IDF(tok))
		}
		if h.charFeat {
			h.addCharGrams(acc, tok, w)
		}
		if h.wordFeat {
			// Whole-token feature in a separate namespace (seeded with a leading
			// 0x00 byte that never begins a token) so it cannot collide with a
			// character n-gram of the same text.
			x := fnv1aByte(h.seed, 0x00)
			h.emit(acc, fnv1aString(x, tok), w)
		}
	}
	return l2normalize(acc)
}

// boundary sentinels wrap a token so that prefix and suffix n-grams are distinct
// features (an "ปลา"-prefix gram differs from the same runes mid-word) and so
// short tokens still yield at least one gram. They are runes that never occur in
// real text.
const (
	gramStart = '\x01'
	gramEnd   = '\x02'
)

// addCharGrams emits every rune n-gram (minN..maxN) of the boundary-wrapped
// token, each weighted by w. Windows are counted per occurrence (a repeated gram
// accumulates), which carries a mild term-frequency signal that normalization
// then bounds. Each n-gram is hashed directly over its rune window — no string
// is materialized, so the hot loop is allocation-free (one []rune per token).
func (h *Hashing) addCharGrams(acc []float32, tok string, w float32) {
	runes := make([]rune, 0, len(tok)+2)
	runes = append(runes, gramStart)
	for _, r := range tok {
		runes = append(runes, r)
	}
	runes = append(runes, gramEnd)
	for n := h.minN; n <= h.maxN; n++ {
		for i := 0; i+n <= len(runes); i++ {
			h.emit(acc, fnv1aRunes(h.seed, runes[i:i+n]), w)
		}
	}
}

// emit maps a feature hash to a (dimension, sign) pair and adds w·sign to that
// dimension. The dimension and the sign are read from different, well-mixed
// regions of the same 64-bit hash so they are effectively independent — the
// signed hash that makes feature hashing an unbiased inner-product estimator.
func (h *Hashing) emit(acc []float32, x uint64, w float32) {
	x ^= x >> 33 // final mix so low and high halves both diffuse
	idx := x % uint64(h.dim)
	if x&(1<<40) == 0 {
		acc[idx] += w
	} else {
		acc[idx] -= w
	}
}

// FNV-1a 64-bit, seedable, computed incrementally. FNV is not cryptographic but
// it mixes short features (character n-grams) cheaply and deterministically,
// which is all feature hashing needs. Hashing a rune window byte-by-byte over
// its UTF-8 encoding is identical to hashing the equivalent string, so the
// mapping is stable regardless of how the feature is presented.
const (
	fnvOffset uint64 = 14695981039346656037
	fnvPrime  uint64 = 1099511628211
)

func fnv1aByte(h uint64, b byte) uint64 { return (h ^ uint64(b)) * fnvPrime }

func fnv1aString(h uint64, s string) uint64 {
	for i := 0; i < len(s); i++ {
		h = fnv1aByte(h, s[i])
	}
	return h
}

func fnv1aRunes(h uint64, runes []rune) uint64 {
	var b [utf8.UTFMax]byte
	for _, r := range runes {
		n := utf8.EncodeRune(b[:], r)
		for i := 0; i < n; i++ {
			h = fnv1aByte(h, b[i])
		}
	}
	return h
}
