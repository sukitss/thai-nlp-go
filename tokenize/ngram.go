package tokenize

// Character n-gram tokenizer — a dictionary-free option that is robust to
// out-of-vocabulary words and typos. It needs no word list, works on any script,
// and degrades gracefully on names/coined terms a word tokenizer would split —
// handy for BM25 and fuzzy/near-duplicate matching.
//
// It slides an n-rune window over each run of non-space characters. Whitespace
// separates runs (an n-gram never spans a space). Runs shorter than n are
// emitted whole.

import "unicode"

// NGram emits character n-grams of size N over non-space runs of text.
// A zero NGram is invalid; use NewNGram.
type NGram struct {
	n int
}

// NewNGram returns a character n-gram tokenizer with window size n (n >= 1).
func NewNGram(n int) *NGram {
	if n < 1 {
		n = 1
	}
	return &NGram{n: n}
}

// Split returns the n-grams of text as strings.
func (g *NGram) Split(text string) []string {
	var out []string
	g.forEach([]rune(text), func(rs []rune) {
		out = append(out, string(rs))
	})
	return out
}

// AppendBytes appends the n-grams of text (joined by sep) to dst and returns the
// extended slice — the zero-allocation-friendly path for indexing.
func (g *NGram) AppendBytes(dst []byte, text string, sep byte) []byte {
	first := true
	g.forEach([]rune(text), func(rs []rune) {
		if !first {
			dst = append(dst, sep)
		}
		first = false
		dst = append(dst, []byte(string(rs))...)
	})
	return dst
}

// forEach calls emit for every n-gram window across non-space runs.
func (g *NGram) forEach(rs []rune, emit func([]rune)) {
	n := len(rs)
	i := 0
	for i < n {
		// skip whitespace
		if unicode.IsSpace(rs[i]) {
			i++
			continue
		}
		// find end of this non-space run
		j := i
		for j < n && !unicode.IsSpace(rs[j]) {
			j++
		}
		run := rs[i:j]
		if len(run) <= g.n {
			emit(run)
		} else {
			for k := 0; k+g.n <= len(run); k++ {
				emit(run[k : k+g.n])
			}
		}
		i = j
	}
}
