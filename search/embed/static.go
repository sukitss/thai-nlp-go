package embed

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/sukitss/thai-nlp-go/multi"
	"github.com/sukitss/thai-nlp-go/vocab"
)

// Static embeds text by mean-pooling PRETRAINED word vectors: it tokenizes the
// text, looks up each token's vector, averages the vectors it finds, and
// L2-normalizes the result. Unlike [Hashing] this gives real distributional
// SEMANTICS — words used in similar contexts have similar vectors, so synonyms
// with no shared characters can still match — but ONLY because a pretrained
// model supplies that knowledge.
//
// # This package ships no model
//
// Static holds only the MECHANISM (parse, store, pool). The vectors come from a
// model the caller loads with [LoadStatic]; nothing is embedded in this package.
// A model's license is the caller's responsibility — a fastText word-vector file
// (https://fasttext.cc) is typically CC-BY-SA, which is a data license, not this
// library's permissive code license. Evaluating a real Thai model end-to-end is
// a follow-up; the loader and pooling are what this package guarantees.
//
// # Storage
//
// Vectors are stored in ONE contiguous []float32 (count·dim) with a word→row map
// into it, so a large model costs one big allocation plus the map, not one small
// slice per word.
//
// # Out-of-vocabulary words
//
// A token with no vector is SKIPPED (documented, and the reason Hashing exists
// for the OOV/typo case). Lookup tries the token verbatim, then its lower-cased
// form, so a lower-cased Latin model still matches capitalized query words. Text
// whose tokens are ALL out-of-vocabulary embeds to an all-zero vector.
type Static struct {
	dim      int
	words    []string       // row → word (for reporting / Len)
	index    map[string]int // word → row
	data     []float32      // flat count·dim, row r at data[r*dim:(r+1)*dim]
	analyzer *multi.Analyzer
	idf      *vocab.Vocab
}

// StaticOption configures a [Static] at load time.
type StaticOption func(*Static)

// WithStaticAnalyzer sets the tokenization front-end. Default: the zero
// [multi.Analyzer].
func WithStaticAnalyzer(a *multi.Analyzer) StaticOption {
	return func(s *Static) { s.analyzer = a }
}

// WithStaticIDF weights each token's vector by the token's BM25 IDF from v when
// pooling, so rare content words dominate the pooled vector over frequent
// function words. nil (the default) uses a plain mean.
func WithStaticIDF(v *vocab.Vocab) StaticOption {
	return func(s *Static) { s.idf = v }
}

// LoadStatic reads pretrained word vectors in the fastText / word2vec TEXT
// format and returns a Static embedder over them:
//
//	<count> <dim>\n            header: number of vectors and their dimension
//	<word> <v1> <v2> … <vdim>\n one line per word (space-separated floats)
//
// The header is required. A word may itself contain spaces (a multi-word entry):
// the LAST dim fields of a line are the vector, everything before them (rejoined
// with single spaces) is the word. Lines with the wrong field count, an
// unparseable float, or a duplicate word are rejected with an error identifying
// the line. The reader is streamed through a buffered scanner, so a multi-GB
// .vec is read without loading the whole file into memory at once.
func LoadStatic(r io.Reader, opts ...StaticOption) (*Static, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024) // long vector lines
	if !sc.Scan() {
		if err := sc.Err(); err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("embed: LoadStatic: empty input (want header \"count dim\")")
	}
	var count, dim int
	if _, err := fmt.Sscanf(strings.TrimSpace(sc.Text()), "%d %d", &count, &dim); err != nil {
		return nil, fmt.Errorf("embed: LoadStatic: bad header %q: %w", sc.Text(), err)
	}
	if dim <= 0 || count < 0 {
		return nil, fmt.Errorf("embed: LoadStatic: bad header count=%d dim=%d", count, dim)
	}
	s := &Static{
		dim:      dim,
		words:    make([]string, 0, count),
		index:    make(map[string]int, count),
		data:     make([]float32, 0, count*dim),
		analyzer: &multi.Analyzer{},
	}
	line := 1
	for sc.Scan() {
		line++
		text := sc.Text()
		if strings.TrimSpace(text) == "" {
			continue
		}
		fields := strings.Fields(text)
		if len(fields) < dim+1 {
			return nil, fmt.Errorf("embed: LoadStatic: line %d has %d fields, want word + %d floats", line, len(fields), dim)
		}
		split := len(fields) - dim
		word := strings.Join(fields[:split], " ")
		if _, dup := s.index[word]; dup {
			return nil, fmt.Errorf("embed: LoadStatic: line %d duplicate word %q", line, word)
		}
		row := len(s.words)
		for _, f := range fields[split:] {
			v, err := strconv.ParseFloat(f, 32)
			if err != nil {
				return nil, fmt.Errorf("embed: LoadStatic: line %d word %q bad float %q: %w", line, word, f, err)
			}
			s.data = append(s.data, float32(v))
		}
		s.index[word] = row
		s.words = append(s.words, word)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	for _, o := range opts {
		o(s)
	}
	if s.analyzer == nil {
		s.analyzer = &multi.Analyzer{}
	}
	return s, nil
}

// Dim reports the embedding dimensionality.
func (s *Static) Dim() int { return s.dim }

// Len reports how many word vectors the model holds.
func (s *Static) Len() int { return len(s.words) }

// Vector returns the raw stored vector for word (no normalization, no copy — do
// not mutate it) and whether the word is in the model. Lookup is exact; callers
// wanting the token/lower-case fallback should use [Static.Embed].
func (s *Static) Vector(word string) ([]float32, bool) {
	if r, ok := s.index[word]; ok {
		return s.data[r*s.dim : (r+1)*s.dim], true
	}
	return nil, false
}

// Embed tokenizes text, mean-pools the in-vocabulary word vectors (IDF-weighted
// if configured), and L2-normalizes the result. Out-of-vocabulary tokens are
// skipped; text with no in-vocabulary token yields an all-zero vector.
func (s *Static) Embed(text string) []float32 {
	out := make([]float32, s.dim)
	var wsum float64
	for _, tok := range s.analyzer.Terms(text) {
		vec, ok := s.lookup(tok)
		if !ok {
			continue
		}
		w := 1.0
		if s.idf != nil {
			w = s.idf.IDF(tok)
		}
		wf := float32(w)
		for i, x := range vec {
			out[i] += x * wf
		}
		wsum += w
	}
	if wsum > 0 {
		inv := float32(1 / wsum)
		for i := range out {
			out[i] *= inv
		}
	}
	return l2normalize(out)
}

// lookup resolves a token to a stored vector, trying the verbatim token then its
// lower-cased form (so a lower-cased Latin model still matches "IT"→"it").
func (s *Static) lookup(tok string) ([]float32, bool) {
	if v, ok := s.Vector(tok); ok {
		return v, true
	}
	if low := strings.ToLower(tok); low != tok {
		return s.Vector(low)
	}
	return nil, false
}
