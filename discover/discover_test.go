package discover_test

import (
	"strings"
	"testing"

	"github.com/sukitss/thai-nlp-go/discover"
)

// words splits a space-delimited sentence — the English shape, where the
// tokenizer is whitespace and the terms discovered are collocations.
func words(s string) []string { return strings.Fields(s) }

func joinSpace(t []string) string { return strings.Join(t, " ") }

func find(terms []discover.Term, text string) (discover.Term, bool) {
	for _, t := range terms {
		if t.Text == text {
			return t, true
		}
	}
	return discover.Term{}, false
}

// A name that always travels together is a term; two common words that merely
// meet often are not. This is the whole job in one test.
func TestCohesionSeparatesNamesFromPhrases(t *testing.T) {
	// The corpus has to be big enough for a name to be rare in it. PMI asks how
	// much likelier this pair is than chance, and in a hundred sentences even a
	// proper name is common — the same reason DefaultMinPMI is set for real
	// prose rather than for examples.
	var docs [][]string
	for i := 0; i < 400; i++ {
		docs = append(docs,
			words("the man at the desk of the school read the letter of the day"),
			words("a woman in the city of the north sent word to the board again"),
			words("they left the top of the hour to the team at the lab today"),
			words("he wrote to the board of the city about the work of the team"),
		)
		if i%40 == 0 { // ten mentions in sixteen hundred sentences
			docs = append(docs,
				words("we study machine learning daily in the lab of the city"),
				words("the report on machine learning went to the board of the school"),
				words("her machine learning course starts at the top of the hour"),
			)
		}
	}
	terms := discover.TermsFromSlices(docs, discover.Options{
		Join:  joinSpace,
		Known: func(string) bool { return true }, // every piece is an ordinary word
	})
	if _, ok := find(terms, "machine learning"); !ok {
		t.Errorf("a phrase whose halves always arrive together is a term:\n%v", terms)
	}
	// "of the" is far more frequent, and means nothing on its own.
	if got, ok := find(terms, "of the"); ok {
		t.Errorf("a frequent function-word pair must not be reported (PMI=%.1f)", got.PMI)
	}
}

// The dictionary's silence is evidence, but it buys a lower entropy bar on one
// side only — a candidate walled in on both sides is part of something larger.
func TestUnknownPiecesEarnALowerBarOnOneSide(t *testing.T) {
	known := func(s string) bool { return s != "zx" && s != "qq" }

	// Free on the left, fixed on the right: a borrowing that is always used in
	// the same phrase. Real, and the point of the concession.
	var oneEdge [][]string
	for _, lead := range []string{"the", "a", "our", "that", "his", "one"} {
		for i := 0; i < 3; i++ {
			oneEdge = append(oneEdge, words(lead+" zx qq arrived"))
		}
	}
	// MaxRun 2 keeps the corpus from also proposing "zx qq arrived", which
	// would swallow this candidate — that is the affix rule's business, tested
	// on its own below.
	oneEdgeOpts := discover.Options{Join: joinSpace, Known: known, MaxRun: 2}
	got, ok := find(discover.TermsFromSlices(oneEdge, oneEdgeOpts), "zx qq")
	if !ok {
		t.Fatal("an unknown pair with one free edge should qualify")
	}
	if !got.Unknown {
		t.Error("the term must be marked as containing an unknown piece")
	}
	if got.RightEntropy != 0 {
		t.Errorf("precondition: the right edge is meant to be fixed, got %.2f", got.RightEntropy)
	}

	// Walled in on both sides: this is a piece of "the zx qq arrived", not a
	// term, and no amount of dictionary silence should rescue it.
	var noEdge [][]string
	for i := 0; i < 18; i++ {
		noEdge = append(noEdge, words("the zx qq arrived"))
	}
	if _, ok := find(discover.TermsFromSlices(noEdge, oneEdgeOpts), "zx qq"); ok {
		t.Error("a candidate with no free edge at all must not qualify")
	}
}

// The rule that cost the headline word when it was written backwards: a
// fragment is one that never stands on its own, not one that is less frequent.
func TestFragmentsGoNotTheirParents(t *testing.T) {
	var docs [][]string
	for i := 0; i < 20; i++ {
		docs = append(docs,
			words("a gob lin walked in"),
			words("the gob lin ran out"),
			words("one gob kit came by"),
			words("that gob kit left now"),
		)
	}
	terms := discover.TermsFromSlices(docs, discover.Options{
		Join: joinSpace, Known: func(s string) bool { return s != "gob" && s != "lin" && s != "kit" },
	})
	if _, ok := find(terms, "gob lin"); !ok {
		t.Errorf("the full name must survive:\n%v", terms)
	}
	if _, ok := find(terms, "gob kit"); !ok {
		t.Errorf("the other full name must survive too:\n%v", terms)
	}
	// "gob" alone is only ever the front of those two.
	if _, ok := find(terms, "gob"); ok {
		t.Error("a shared prefix that never stands alone must be dropped")
	}
}

// Valid is the caller's language rule, and it must be able to veto.
func TestValidCanVetoOnLanguageGrounds(t *testing.T) {
	var docs [][]string
	for i := 0; i < 20; i++ {
		docs = append(docs, words("aa bb cc"), words("xx aa bb yy"), words("zz aa bb qq"))
	}
	opts := discover.Options{Join: joinSpace}
	if _, ok := find(discover.TermsFromSlices(docs, opts), "aa bb"); !ok {
		t.Fatal("precondition: the pair should qualify without a language rule")
	}
	opts.Valid = func(string) bool { return false }
	if terms := discover.TermsFromSlices(docs, opts); len(terms) != 0 {
		t.Errorf("Valid must be able to reject everything, got %v", terms)
	}
}

// Nothing to learn from nothing.
func TestEmptyAndTinyCorporaAreQuiet(t *testing.T) {
	if got := discover.TermsFromSlices(nil, discover.Options{}); len(got) != 0 {
		t.Errorf("empty corpus → no terms, got %v", got)
	}
	one := [][]string{words("a b c")}
	if got := discover.TermsFromSlices(one, discover.Options{Join: joinSpace}); len(got) != 0 {
		t.Errorf("a single sentence cannot meet MinCount, got %v", got)
	}
}

// Scores travel with the term so a caller can re-rank or explain a proposal.
func TestTermsCarryTheirEvidence(t *testing.T) {
	var docs [][]string
	for i := 0; i < 12; i++ {
		docs = append(docs, words("alpha beta one"), words("two alpha beta three"), words("four alpha beta"))
	}
	terms := discover.TermsFromSlices(docs, discover.Options{Join: joinSpace})
	got, ok := find(terms, "alpha beta")
	if !ok {
		t.Fatalf("expected the pair, got %v", terms)
	}
	if got.Count != 36 {
		t.Errorf("count = %d, want 36", got.Count)
	}
	if got.PMI <= 0 || got.LeftEntropy <= 0 || got.RightEntropy <= 0 {
		t.Errorf("evidence must be reported, got %+v", got)
	}
	if len(got.Tokens) != 2 {
		t.Errorf("tokens must be kept, got %v", got.Tokens)
	}
}

// A term used in one fixed phrase is otherwise swallowed by that phrase. The
// wrapping word gets to swallow it only if it belongs to the name rather than
// merely describing it, and living elsewhere in the corpus is the proof.
func TestAffixMustLiveOutsideTheTermItWraps(t *testing.T) {
	known := func(s string) bool { return s != "zx" && s != "qq" }
	build := func(elsewhere int) [][]string {
		var docs [][]string
		for _, lead := range []string{"the", "a", "our", "that", "his", "one"} {
			for i := 0; i < 3; i++ {
				docs = append(docs, words(lead+" zx qq armoured"))
			}
		}
		for i := 0; i < elsewhere; i++ {
			docs = append(docs, words("the armoured door of the hall was shut"))
		}
		return docs
	}
	opts := discover.Options{Join: joinSpace, Known: known, Affix: known}

	// "armoured" is a word of its own here: it describes the creature.
	free := discover.TermsFromSlices(build(40), opts)
	if _, ok := find(free, "zx qq armoured"); ok {
		t.Errorf("the wrapping phrase should not be reported when the wrapper is ordinary:\n%v", free)
	}
	if _, ok := find(free, "zx qq"); !ok {
		t.Errorf("the name should survive the phrase it is always used in:\n%v", free)
	}

	// "armoured" now occurs almost only in this phrase: it reads as part of the
	// name, and the name is the whole phrase.
	if _, ok := find(discover.TermsFromSlices(build(0), opts), "zx qq armoured"); !ok {
		t.Error("a wrapper with no life of its own belongs to the term")
	}

	// Without an Affix function a caller gets the conservative reading.
	plain := discover.Options{Join: joinSpace, Known: known}
	if _, ok := find(discover.TermsFromSlices(build(40), plain), "zx qq"); ok {
		t.Error("with no Affix rule, the longer candidate must win")
	}
}
