// Package discover finds the terms a corpus uses that its dictionary does not
// know: names, loanwords, jargon, and multi-word expressions that belong
// together.
//
// The problem it solves is visible in any dictionary-driven pipeline. A Thai
// segmenter meets "ก็อบลิน" (goblin), cannot find it, and emits ก็·อบ·ลิ·น —
// four fragments, none of them the word. The text is then indexed under
// fragments, and a search for the word itself returns nothing although the
// corpus says it 149 times. English has the same gap with a different shape:
// "machine learning" is two known words whose meaning lives in the pair.
//
// # How it decides
//
// A candidate is a short run of adjacent tokens. It becomes a term when the
// corpus itself vouches for it, three ways:
//
//   - Frequency. Seen often enough not to be an accident.
//   - Branching entropy. A real term appears in varied company on BOTH sides;
//     a fragment is followed by the same neighbour nearly every time, because
//     it only exists as part of something longer.
//   - Cohesion (PMI). Its pieces occur together far more often than their
//     individual rates would predict — the signal that separates a name from
//     two ordinary words that happen to be adjacent.
//
// Two structural rules then clean up what statistics alone leaves behind: a
// candidate seen almost exclusively inside a longer candidate is a fragment of
// it (see Terms), and the caller may reject candidates that are not
// well-formed in its language (Options.Valid).
//
// # Language independence
//
// Nothing here reads characters. Input is a sequence of already-tokenized
// documents, so whatever produced those tokens decides the language: newmm for
// Thai, whitespace for English, multi.Segment for mixed text. In Thai the runs
// are fragments the segmenter could not resolve; in English they are
// collocations. The arithmetic is the same.
package discover

import "math"

// Options tunes what counts as a term. The zero value is usable but strict-ish;
// see the DefaultXxx constants for what each field defaults to when left zero.
type Options struct {
	// MinCount is how often a candidate must appear. Default DefaultMinCount.
	MinCount int
	// MinEntropy is the minimum branching entropy (bits) required on EACH side.
	// Default DefaultMinEntropy.
	MinEntropy float64
	// MinEntropyUnknown is the bar for candidates containing a token the
	// dictionary does not know. It is lower on purpose: the dictionary's own
	// silence is evidence, and a rare loanword may appear a dozen times in the
	// same phrase without ever earning entropy — "ทานูกิ" occurs 19 times in a
	// novel corpus, always followed by the same word. Default
	// DefaultMinEntropyUnknown.
	MinEntropyUnknown float64
	// MinPMI is the cohesion floor applied to candidates whose pieces are all
	// known words — the case where frequency proves nothing, since common words
	// are common. Default DefaultMinPMI.
	//
	// Pick it from the corpus rather than from taste: score a handful of known
	// phrases and known names and put the floor in the gap between them. On a
	// Thai novel corpus, ordinary phrases scored 1.3-5.7 and proper names
	// 13-24.6, so 9 sat comfortably between.
	MinPMI float64
	// MinPMIUnknown is the cohesion floor for candidates containing a token the
	// dictionary cannot name. It is far below MinPMI — the dictionary's silence
	// is most of the evidence for those — but not zero, because the debris a
	// mis-segmented name leaves behind is unknown too: "เฟียที่" scores 2.4 and
	// is the tail of a name plus a preposition. Default DefaultMinPMIUnknown.
	MinPMIUnknown float64
	// MaxRun is the longest run of tokens considered. Default DefaultMaxRun.
	MaxRun int
	// Known reports whether a token is already a dictionary word. When set, a
	// candidate need only clear MinPMIUnknown rather than MinPMI — which
	// is what stops "of the" and "ของผม" from being reported as discoveries.
	// nil means nothing is known, and every run is a candidate.
	Known func(token string) bool
	// Valid rejects candidates that are not well-formed in the caller's
	// language — Thai orthography, say, where a run of bare consonants cannot
	// be pronounced. nil accepts everything.
	Valid func(term string) bool
	// Join builds the surface form from its tokens. nil concatenates directly,
	// which is right for scripts without word spacing; pass a space-joiner for
	// English.
	Join func(tokens []string) string
	// Affix reports whether a token is ordinary vocabulary that can be peeled
	// off a term without changing what the term names — an adjective, a title,
	// a classifier. It is asked only about the tokens a LONGER candidate adds
	// around a shorter one, and it decides which of the two is the term:
	// "ทานูกิ" survives inside "ทานูกิหุ้มเกราะ" because "หุ้ม" and "เกราะ"
	// are ordinary words, while "หนิงเอ๋อร์" does not survive inside
	// "เซียวหนิงเอ๋อร์" because "เซียว" is a surname that occurs 209 times in
	// that corpus and 201 of them are this one name.
	//
	// Being ordinary vocabulary is not enough on its own; the token must also
	// be shown to live outside the term (see AffixFreeShare), which is what
	// separates those two cases. Nil means longer candidates always win, which
	// is the safe reading when the caller has no vocabulary to consult.
	Affix func(token string) bool
	// AffixFreeShare is the share of an affix token's occurrences that must
	// fall outside the term it wraps, before it is believed to be incidental to
	// it rather than part of it. Default DefaultAffixFreeShare.
	AffixFreeShare float64
	// TokenFilter drops tokens before any of this runs — punctuation, digits,
	// stop words. nil keeps everything.
	TokenFilter func(token string) bool
}

// Defaults for Options. They are exported because a caller that overrides one
// value should be able to see what it is departing from.
const (
	DefaultMinCount   = 5
	DefaultMinEntropy = 1.5
	// DefaultMinEntropyUnknown is the bar for candidates the dictionary cannot
	// name, where only the freer side has to clear it — see the asymmetry note
	// on MinEntropyUnknown.
	DefaultMinEntropyUnknown = 0.2
	// DefaultAffixFreeShare requires a quarter of an affix's occurrences to fall
	// outside the term it wraps.
	//
	// Measured on a novel corpus, where the two cases sit surprisingly close in
	// absolute terms and far apart in relative ones: "หุ้ม" occurs 28 times, 19
	// of them wrapping "ทานูกิ", so a third of its life is its own — it is a
	// word that happens to describe this creature. "เซียว" occurs 209 times,
	// 201 of them inside "เซียวหนิงเอ๋อร์", so it has almost no life apart from
	// that name — it is part of it. Counting the share rather than the ratio is
	// what keeps this from depending on how common the term itself is.
	DefaultAffixFreeShare = 0.25
	DefaultMinPMI         = 9.0
	// DefaultMinPMIUnknown asks an unknown candidate to be some 32x likelier
	// than chance. Measured on a novel corpus, that is the gap between real
	// borrowings (ทานูกิ 24.5, ก็อบลิน 21.9, เสิ่นซิ่ว 8.9) and the debris
	// around a mis-segmented name (เฟียที่ 2.4, เฟียก็ 1.4, กับเฟีย 4.0).
	DefaultMinPMIUnknown = 5.0
	DefaultMaxRun        = 4
)

// join returns the configured joiner, or concatenation — the Thai default,
// where a word boundary is not written.
func (o Options) join() func([]string) string {
	if o.Join != nil {
		return o.Join
	}
	return concat
}

func (o Options) withDefaults() Options {
	if o.MinCount <= 0 {
		o.MinCount = DefaultMinCount
	}
	if o.MinEntropy <= 0 {
		o.MinEntropy = DefaultMinEntropy
	}
	if o.MinEntropyUnknown <= 0 {
		o.MinEntropyUnknown = DefaultMinEntropyUnknown
	}
	if o.AffixFreeShare <= 0 {
		o.AffixFreeShare = DefaultAffixFreeShare
	}
	if o.MinPMI <= 0 {
		o.MinPMI = DefaultMinPMI
	}
	if o.MinPMIUnknown <= 0 {
		o.MinPMIUnknown = DefaultMinPMIUnknown
	}
	if o.MaxRun <= 0 {
		o.MaxRun = DefaultMaxRun
	}
	return o
}

// Term is one discovered term and the evidence behind it. The scores are
// reported, not just thresholded, so a caller can re-rank, set its own cut-off,
// or show a person why something was proposed.
type Term struct {
	// Text is the surface form, built by Options.Join.
	Text string
	// Tokens are the pieces it was assembled from.
	Tokens []string
	// Count is how many times the run occurred.
	Count int
	// LeftEntropy and RightEntropy are the branching entropies, in bits.
	LeftEntropy, RightEntropy float64
	// PMI is pointwise mutual information against the product of its pieces'
	// rates: how much more often they appear together than chance predicts.
	PMI float64
	// Unknown is true when at least one piece was not a known word.
	Unknown bool
}

// candidate accumulates evidence while scanning.
type candidate struct {
	tokens  []string
	count   int
	unknown bool
	left    map[string]int
	right   map[string]int
}

// Terms scans tokenized documents and returns the terms the corpus vouches for,
// most frequent first.
//
// docs is called once per document and must yield that document's tokens in
// order; passing a slice of slices is the common case (see TermsFromSlices).
// Memory is proportional to the number of distinct candidates, not to the
// corpus size.
func Terms(docs func(yield func(tokens []string) bool), opts Options) []Term {
	o := opts.withDefaults()
	join := o.join()
	known := o.Known
	if known == nil {
		known = func(string) bool { return false }
	}

	// concat is the common joiner and the one worth a fast path.
	fastKey := o.Join == nil
	var key []byte

	cands := map[string]*candidate{}
	unigram := map[string]int{}
	total := 0

	docs(func(toks []string) bool {
		if o.TokenFilter != nil {
			kept := toks[:0:0]
			for _, t := range toks {
				if o.TokenFilter(t) {
					kept = append(kept, t)
				}
			}
			toks = kept
		}
		for _, t := range toks {
			unigram[t]++
			total++
		}
		for i := range toks {
			for n := 2; n <= o.MaxRun && i+n <= len(toks); n++ {
				run := toks[i : i+n]
				// Look the candidate up without allocating its text: for a map
				// indexed by a []byte conversion the compiler skips the copy,
				// and only a candidate seen for the first time needs a key of
				// its own. Most n-grams in a corpus have been seen before, so
				// this is most of the work.
				var c *candidate
				var text string
				if fastKey {
					key = appendConcat(key[:0], run)
					if len(key) == 0 {
						continue
					}
					c = cands[string(key)]
				} else {
					text = join(run)
					if text == "" {
						continue
					}
					c = cands[text]
				}
				if c == nil {
					if fastKey {
						text = string(key)
					}
					unknown := false
					for _, p := range run {
						if !known(p) {
							unknown = true
							break
						}
					}
					c = &candidate{
						tokens:  append([]string(nil), run...),
						unknown: unknown,
						left:    map[string]int{},
						right:   map[string]int{},
					}
					cands[text] = c
				}
				c.count++
				if i > 0 {
					c.left[toks[i-1]]++
				}
				if i+n < len(toks) {
					c.right[toks[i+n]]++
				}
			}
		}
		return true
	})

	var out []Term
	for text, c := range cands {
		if term, ok := judge(text, c, unigram, total, o); ok {
			out = append(out, term)
		}
	}
	return dropFragments(sortByCount(out), unigram, o)
}

// judge applies the four signals to one candidate.
func judge(text string, c *candidate, unigram map[string]int, total int, o Options) (Term, bool) {
	if c.count < o.MinCount {
		return Term{}, false
	}
	if o.Valid != nil && !o.Valid(text) {
		return Term{}, false
	}
	hl, hr := entropy(c.left), entropy(c.right)
	// Entropy locates the EDGES of a unit: a real one is free to take different
	// neighbours, a fragment is stuck to whatever completes it. A phrase of
	// ordinary words must prove both edges, or every window sliding across a
	// common sentence qualifies. A candidate the dictionary cannot name only
	// has to prove the freer edge — a rare borrowing can be a word and still
	// keep the same company every time it is used, and demanding both edges
	// loses it (measured: "ทานูกิ", 19 occurrences, always followed by the same
	// word, right entropy 0).
	free, stuck := hl, hr
	if hr > hl {
		free, stuck = hr, hl
	}
	if c.unknown {
		if free < o.MinEntropyUnknown {
			return Term{}, false
		}
		// An edge with exactly one neighbour every time is not a boundary the
		// corpus found; it is one this candidate stopped short of. Whether that
		// matters depends on what the neighbour is. "ทานูกิ" is always followed
		// by "หุ้ม", an ordinary word — the unit is complete and extending it
		// would only produce a phrase. "นูกิหุ้มเกราะ" is always preceded by
		// "ทา", which is not a word at all — the unit starts in the middle of a
		// name, and extending it is exactly what it needs.
		if !edgeSettled(c.left, unigram, o) || !edgeSettled(c.right, unigram, o) {
			return Term{}, false
		}
	} else if stuck < o.MinEntropy {
		return Term{}, false
	}
	pmi := cohesion(c, unigram, total)
	// A run made entirely of known words is usually just a phrase; it earns a
	// place only by clinging together far harder than chance.
	floor := o.MinPMI
	if c.unknown {
		floor = o.MinPMIUnknown
	}
	if pmi < floor {
		return Term{}, false
	}
	return Term{
		Text: text, Tokens: c.tokens, Count: c.count,
		LeftEntropy: hl, RightEntropy: hr, PMI: pmi, Unknown: c.unknown,
	}, true
}

// TermsFromSlices is Terms over an in-memory corpus.
func TermsFromSlices(docs [][]string, opts Options) []Term {
	return Terms(func(yield func([]string) bool) {
		for _, d := range docs {
			if !yield(d) {
				return
			}
		}
	}, opts)
}

func concat(tokens []string) string {
	return string(appendConcat(nil, tokens))
}

func appendConcat(dst []byte, tokens []string) []byte {
	for _, t := range tokens {
		dst = append(dst, t...)
	}
	return dst
}

// edgeDominance is the share of one neighbour that makes an edge look decided
// in advance rather than found. Nine in ten leaves room for a name that appears
// once or twice outside its usual phrase — measured: "กิหุ้มเกราะ" is preceded
// by "นู" in 19 of its 21 occurrences, and it is a fragment of ทานูกิหุ้มเกราะ.
const edgeDominance = 0.9

// edgeSettled reports whether an edge is a boundary the corpus agrees on.
//
// An edge with varied neighbours is settled by definition. One dominated by a
// single neighbour is settled only when that neighbour is ordinary vocabulary,
// which means the candidate ends where a word ends rather than where a longer
// name was cut in half.
func edgeSettled(neighbours map[string]int, unigram map[string]int, o Options) bool {
	if o.Affix == nil || len(neighbours) == 0 {
		return true
	}
	total, top, topToken := 0, 0, ""
	for token, n := range neighbours {
		total += n
		if n > top {
			top, topToken = n, token
		}
	}
	if float64(top) < edgeDominance*float64(total) {
		return true
	}
	return allAffix([]string{topToken}, unigram, top, o)
}

// entropy is Shannon entropy over the neighbour distribution, in bits.
func entropy(m map[string]int) float64 {
	total := 0
	for _, n := range m {
		total += n
	}
	if total == 0 {
		return 0
	}
	h := 0.0
	for _, n := range m {
		p := float64(n) / float64(total)
		h -= p * math.Log2(p)
	}
	return h
}

// cohesion is PMI between the run and the product of its pieces.
func cohesion(c *candidate, unigram map[string]int, total int) float64 {
	if total == 0 || len(c.tokens) == 0 {
		return 0
	}
	p := float64(c.count) / float64(total)
	independent := 1.0
	for _, t := range c.tokens {
		independent *= float64(unigram[t]) / float64(total)
	}
	if independent <= 0 || p <= 0 {
		return 0
	}
	return math.Log2(p / independent)
}
