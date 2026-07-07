package keyword

import (
	"math"
	"strings"
	"unicode"
)

// YAKE scores single words with the corpus-free statistical weighting of YAKE!
// (Campos, Mangaravite, Pasquali, Jorge, Nunes & Jatowt, 2020). No corpus and no
// language model: each word's importance follows from five features computed
// from the document alone —
//
//   - Casing: how often the word appears as an ACRONYM (all-caps) or capitalized;
//     acronyms and proper-noun casing signal importance. This is where "IT"/"POS"
//     earn weight rather than being discarded.
//   - Position: words that appear earlier in the document weigh more.
//   - Frequency: normalized against the document's mean/σ term frequency.
//   - Relatedness: words that co-occur with MANY different neighbours behave like
//     stop words and are penalized.
//   - Dispersion: words spread across more sentences weigh more.
//
// YAKE's raw term weight is a COST — lower is more important. Extract inverts it
// so that, like every extractor here, a LARGER [Keyword.Score] is better. This
// is the unigram form (n=1); YAKE's n-gram chaining is a documented extension
// point (see the package-level notes). The [Keyword.Text] is the acronym-aware
// folded key.
type YAKE struct {
	cfg config
}

// NewYAKE returns a corpus-free YAKE extractor.
func NewYAKE(opts ...Option) *YAKE { return &YAKE{cfg: defaultConfig().apply(opts)} }

// yakeStat accumulates one word's per-document statistics.
type yakeStat struct {
	tf        int
	tfUpper   int // occurrences written as an all-caps acronym
	tfCapital int // occurrences capitalized (first letter upper, not all-caps)
	left      map[string]struct{}
	right     map[string]struct{}
	leftN     int // occurrences that had a content-word neighbour on the left
	rightN    int
	sentences map[int]struct{} // sentence indices the word appears in
	offsets   []int            // 0-based content-word positions (for median)
}

// Extract returns the k words with the best (inverted) YAKE weight, best first.
func (e *YAKE) Extract(text string, k int) []Keyword {
	if k <= 0 {
		return nil
	}
	_, terms := e.cfg.analyze(text)

	stats := map[string]*yakeStat{}
	get := func(key string) *yakeStat {
		s := stats[key]
		if s == nil {
			s = &yakeStat{left: map[string]struct{}{}, right: map[string]struct{}{}, sentences: map[int]struct{}{}}
			stats[key] = s
		}
		return s
	}

	sentence := 0
	pos := 0          // running content-word index
	prevContent := "" // previous content word (for right-neighbour linkage)
	prevIsContent := false
	for _, t := range terms {
		if t.sent { // sentence terminator dropped into the gap before this token
			sentence++
		}
		if t.kind != kindContent {
			if isSentenceBreak(t.surface) {
				sentence++
			}
			prevIsContent = false
			continue
		}
		s := get(t.key)
		s.tf++
		s.sentences[sentence] = struct{}{}
		s.offsets = append(s.offsets, pos)
		if acr := isAcronymSurface(t.surface); acr {
			s.tfUpper++
		} else if isCapitalized(t.surface) {
			s.tfCapital++
		}
		if prevIsContent {
			// prevContent is on this word's left; this word is on prev's right.
			s.left[prevContent] = struct{}{}
			s.leftN++
			ps := stats[prevContent]
			ps.right[t.key] = struct{}{}
			ps.rightN++
		}
		prevContent = t.key
		prevIsContent = true
		pos++
	}
	if len(stats) == 0 {
		return nil
	}

	// Corpus-level (document) frequency moments over distinct terms.
	var sum, sumSq float64
	maxTF := 1
	for _, s := range stats {
		sum += float64(s.tf)
		sumSq += float64(s.tf) * float64(s.tf)
		if s.tf > maxTF {
			maxTF = s.tf
		}
	}
	nTerms := float64(len(stats))
	meanTF := sum / nTerms
	varTF := sumSq/nTerms - meanTF*meanTF
	if varTF < 0 {
		varTF = 0
	}
	stdTF := math.Sqrt(varTF)
	nSent := sentence + 1

	kws := make([]Keyword, 0, len(stats))
	for key, s := range stats {
		wCase := float64(max(s.tfUpper, s.tfCapital)) / (1 + math.Log(float64(s.tf)))
		wPos := math.Log(math.Log(3 + medianFloat(s.offsets)))
		wFreq := float64(s.tf) / (meanTF + stdTF + 1e-12)
		dl := 0.0
		if s.leftN > 0 {
			dl = float64(len(s.left)) / float64(s.leftN)
		}
		dr := 0.0
		if s.rightN > 0 {
			dr = float64(len(s.right)) / float64(s.rightN)
		}
		wRel := 1 + (dl+dr)*float64(s.tf)/float64(maxTF)
		wSpread := float64(len(s.sentences)) / float64(nSent)

		raw := (wRel * wPos) / (wCase + wFreq/wRel + wSpread/wRel)
		// raw is a cost (lower = better) and > 0; invert for larger-is-better.
		kws = append(kws, Keyword{Text: key, Score: 1 / (raw + 1e-9)})
	}
	return topK(kws, k)
}

// isSentenceBreak reports whether a delimiter surface ends a sentence.
func isSentenceBreak(surface string) bool {
	return strings.ContainsAny(surface, ".!?\n\r。！？")
}

// isAcronymSurface reports whether a token surface is an all-caps acronym.
func isAcronymSurface(surface string) bool {
	_, acr := fold(surface)
	return acr
}

// isCapitalized reports whether a surface starts with an uppercase letter (and
// is not an all-caps acronym, which is counted separately).
func isCapitalized(surface string) bool {
	for _, r := range surface {
		if unicode.IsLetter(r) {
			return unicode.IsUpper(r)
		}
	}
	return false
}

// medianFloat returns the median of a small non-empty slice of positions, or 0
// for an empty slice (so ln(ln(3+0)) is defined).
func medianFloat(xs []int) float64 {
	n := len(xs)
	if n == 0 {
		return 0
	}
	// offsets are appended in increasing order, so xs is already sorted.
	if n%2 == 1 {
		return float64(xs[n/2])
	}
	return (float64(xs[n/2-1]) + float64(xs[n/2])) / 2
}
