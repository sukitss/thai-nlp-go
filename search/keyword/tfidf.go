package keyword

import (
	"bytes"

	"github.com/sukitss/thai-nlp-go/vocab"
)

// TFIDFTopK scores each content word by tf(term in this doc) × idf(term from the
// corpus) and returns the top-k. It is the simplest, fastest extractor and the
// reference the others are compared against.
//
// The idf comes from a supplied [vocab.Vocab] — a corpus vocabulary carrying
// per-term document frequency, from which vocab.IDF gives the BM25 inverse
// document frequency. That corpus is what lets tf-idf tell a document-specific
// word (rare across the corpus, high idf) from a globally common one that
// happens to repeat in this document (e.g. "ติดต่อ", high df, low idf): raw term
// frequency alone cannot. The vocabulary's terms MUST be built with the same
// acronym-aware casing this package produces (fold ordinary words to lower case,
// keep acronyms verbatim) or the idf lookup will miss; the easiest way is to
// feed vocab.Builder the keys this package would produce (see BuildVocab).
//
// A nil vocab makes idf ≡ 1, degrading to a pure term-frequency ranking — still
// a usable within-document salience for a single document, just without the
// corpus signal that suppresses globally common words.
type TFIDFTopK struct {
	cfg   config
	vocab *vocab.Vocab
}

// NewTFIDFTopK returns a tf-idf extractor drawing idf from v (nil ⇒ pure tf).
func NewTFIDFTopK(v *vocab.Vocab, opts ...Option) *TFIDFTopK {
	return &TFIDFTopK{cfg: defaultConfig().apply(opts), vocab: v}
}

// Extract returns the k content words with the highest tf-idf, best first.
func (e *TFIDFTopK) Extract(text string, k int) []Keyword {
	if k <= 0 {
		return nil
	}
	_, terms := e.cfg.analyze(text)
	tf := map[string]int{}
	for _, t := range terms {
		if t.kind == kindContent {
			tf[t.key]++
		}
	}
	if len(tf) == 0 {
		return nil
	}
	kws := make([]Keyword, 0, len(tf))
	for key, f := range tf {
		idf := 1.0
		if e.vocab != nil {
			idf = e.vocab.IDF(key)
		}
		kws = append(kws, Keyword{Text: key, Score: float64(f) * idf})
	}
	return topK(kws, k)
}

// BuildVocab is a convenience that builds a corpus vocabulary from raw documents
// using EXACTLY the analysis a keyword extractor applies (same tokenization,
// same acronym-aware folding, same drops), so idf lookups line up with the keys
// TFIDFTopK produces. Each document contributes its distinct content-word keys
// to document frequency. Pass the same options you will give NewTFIDFTopK.
//
// It is a helper, not policy: callers who already maintain a vocab.Vocab over
// their corpus (with matching casing) should pass that instead.
func BuildVocab(docs []string, opts ...Option) *vocab.Vocab {
	cfg := defaultConfig().apply(opts)
	b := vocab.NewBuilder()
	for _, d := range docs {
		_, terms := cfg.analyze(d)
		b.AddDoc(contentKeys(terms))
	}
	// vocab exposes no direct Builder→Vocab; round-trip through its versioned
	// snapshot format (small, in-memory) to obtain the immutable *Vocab.
	var buf bytes.Buffer
	if err := b.Save(&buf); err != nil {
		return nil // Save to a bytes.Buffer cannot fail; guard for the type only
	}
	v, err := vocab.Load(&buf)
	if err != nil {
		return nil
	}
	return v
}
