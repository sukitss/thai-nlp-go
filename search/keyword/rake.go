package keyword

// RAKE is Rapid Automatic Keyword Extraction (Rose, Engel, Cramer & Cowley,
// 2010). It is corpus-free and captures MULTI-WORD keyphrases, which the
// single-word extractors cannot:
//
//   - Candidate phrases are the maximal runs of content words between delimiters
//     (stop words, punctuation, dropped numbers). "ปลา IT, พี่ปลา" yields the
//     candidate phrases "ปลา IT", "พี่ปลา"; "ศึกชิงเจ้าสำนัก" (no stop word inside)
//     stays one phrase.
//   - Each distinct word w scores deg(w)/freq(w), where freq(w) is how many
//     phrase slots it fills and deg(w) is the total length of the phrases it
//     appears in (its co-occurrence degree, self included). Words that recur
//     inside longer phrases score highest.
//   - A phrase scores the sum of its words' scores, so longer salient phrases
//     outrank their component unigrams.
//
// The returned [Keyword.Text] is the exact original substring spanning the
// phrase (from the normalized text), so casing and internal spacing survive:
// "ปลา IT" keeps its acronym and its space, "ปลาit" (stuck) keeps its lack of
// one. Duplicate phrases collapse to one entry (they share a score).
type RAKE struct {
	cfg config
}

// NewRAKE returns a corpus-free RAKE extractor.
func NewRAKE(opts ...Option) *RAKE { return &RAKE{cfg: defaultConfig().apply(opts)} }

// phrase is a candidate keyphrase: its content-word keys plus the byte span of
// its surface form in the normalized text.
type phrase struct {
	keys  []string
	start int
	end   int
}

// Extract returns the k highest-scoring candidate phrases, best first.
func (e *RAKE) Extract(text string, k int) []Keyword {
	if k <= 0 {
		return nil
	}
	norm, terms := e.cfg.analyze(text)
	phrases := candidatePhrases(terms)
	if len(phrases) == 0 {
		return nil
	}

	// Word co-occurrence statistics over the candidate phrases.
	freq := map[string]int{}
	deg := map[string]int{}
	for _, p := range phrases {
		n := len(p.keys)
		for _, w := range p.keys {
			freq[w]++
			deg[w] += n // degree includes the word itself (self co-occurrence)
		}
	}
	score := make(map[string]float64, len(freq))
	for w, f := range freq {
		score[w] = float64(deg[w]) / float64(f)
	}

	// One entry per distinct phrase surface (duplicates share a score).
	seen := make(map[string]struct{}, len(phrases))
	kws := make([]Keyword, 0, len(phrases))
	for _, p := range phrases {
		surface := norm[p.start:p.end]
		if _, dup := seen[surface]; dup {
			continue
		}
		seen[surface] = struct{}{}
		var s float64
		for _, w := range p.keys {
			s += score[w]
		}
		kws = append(kws, Keyword{Text: surface, Score: s})
	}
	return topK(kws, k)
}

// candidatePhrases splits the token stream into maximal runs of content words,
// breaking at every delimiter. Each phrase records the byte span from its first
// word's start to its last word's end, so the surface can be sliced out whole.
func candidatePhrases(terms []term) []phrase {
	var out []phrase
	var cur phrase
	flush := func() {
		if len(cur.keys) > 0 {
			out = append(out, cur)
			cur = phrase{}
		}
	}
	for _, t := range terms {
		if t.kind != kindContent {
			flush()
			continue
		}
		if t.hard { // dropped punctuation in the gap before this word
			flush()
		}
		if len(cur.keys) == 0 {
			cur.start = t.start
		}
		cur.keys = append(cur.keys, t.key)
		cur.end = t.end
	}
	flush()
	return out
}
