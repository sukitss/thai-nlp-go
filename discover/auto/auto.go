// Package auto is the one-call entry point for term discovery: give it raw
// documents in any mix of scripts and it routes every decision — tokenizer,
// dictionary, spelling rule, how tokens join back into a term — by the script
// each candidate is written in.
//
//	terms := auto.Terms(docs, auto.Options{})
//
// Routing is the point. The tests a term must pass are not the same in every
// script: Thai has an embedded dictionary and a rule for which letter sequences
// can be read aloud, English has neither but writes its word boundaries down.
// Sending Thai text through the English rules, or the reverse, quietly returns
// nothing — which is what a caller gets from mixed corpora otherwise.
//
// It is a separate package for the same reason multi is: routing pulls in every
// language's tokenizer, and a caller with only Thai text should not pay for the
// rest. Use discover with discover/thai directly when the script is known in
// advance.
package auto

import (
	"strings"
	"unicode"

	"github.com/sukitss/thai-nlp-go/discover"
	"github.com/sukitss/thai-nlp-go/discover/thai"
	"github.com/sukitss/thai-nlp-go/multi"
)

// Options is discover.Options; any field left zero keeps the routed default,
// and any field set replaces it for every script at once.
type Options = discover.Options

// minAffixRunes mirrors discover/thai: a word shorter than this is a particle
// or a preposition, and peeling it off a term is how "อบลิน" and "he Smith"
// happen.
const minAffixRunes = 3

// Terms tokenizes each document with multi.Segment and returns the terms the
// corpus vouches for, most frequent first.
func Terms(docs []string, opts Options) []discover.Term {
	base := routed()
	if opts.MinCount > 0 {
		base.MinCount = opts.MinCount
	}
	if opts.MinEntropy > 0 {
		base.MinEntropy = opts.MinEntropy
	}
	if opts.MinEntropyUnknown > 0 {
		base.MinEntropyUnknown = opts.MinEntropyUnknown
	}
	if opts.MinPMI > 0 {
		base.MinPMI = opts.MinPMI
	}
	if opts.MaxRun > 0 {
		base.MaxRun = opts.MaxRun
	}
	if opts.AffixFreeShare > 0 {
		base.AffixFreeShare = opts.AffixFreeShare
	}
	if opts.Known != nil {
		base.Known = opts.Known
	}
	if opts.Valid != nil {
		base.Valid = opts.Valid
	}
	if opts.Affix != nil {
		base.Affix = opts.Affix
	}
	if opts.Join != nil {
		base.Join = opts.Join
	}
	if opts.TokenFilter != nil {
		base.TokenFilter = opts.TokenFilter
	}
	return discover.Terms(func(yield func([]string) bool) {
		for _, d := range docs {
			if !yield(multi.Segment(d)) {
				return
			}
		}
	}, base)
}

// routed builds the options that dispatch on script.
func routed() discover.Options {
	th := thai.Options()
	return discover.Options{
		Known: func(token string) bool {
			if !isThai(token) {
				// No dictionary for this script. Treating every word as known
				// is the conservative reading: an unknown word is allowed to
				// skip the cohesion test, and claiming that for a whole script
				// on no evidence would report every common phrase in it.
				return true
			}
			return th.Known != nil && th.Known(token)
		},
		Valid: func(text string) bool {
			if isThai(text) {
				return thai.Pronounceable(text)
			}
			return spellsAWord(text)
		},
		Affix: func(token string) bool {
			if len([]rune(token)) < minAffixRunes {
				return false
			}
			if isThai(token) {
				return th.Affix != nil && th.Affix(token)
			}
			// Elsewhere, ordinary vocabulary is anything that spells a word;
			// what actually keeps a name's own halves from being peeled off is
			// AffixFreeShare, which asks the corpus rather than a dictionary.
			return spellsAWord(token)
		},
		Join: joinByScript,
	}
}

// joinByScript writes tokens back out the way the script writes them: Thai and
// the CJK scripts run words together, the alphabetic ones separate them. A term
// is compared and reported as text, so getting this wrong turns "machine
// learning" into "machinelearning" and never matches anything a caller has.
func joinByScript(tokens []string) string {
	var b strings.Builder
	for i, t := range tokens {
		if i > 0 && spaced(tokens[i-1]) && spaced(t) {
			b.WriteByte(' ')
		}
		b.WriteString(t)
	}
	return b.String()
}

func spaced(token string) bool {
	for _, r := range token {
		if unicode.IsLetter(r) && !isThaiRune(r) && !unicode.Is(unicode.Han, r) &&
			!unicode.Is(unicode.Hiragana, r) && !unicode.Is(unicode.Katakana, r) {
			return true
		}
	}
	return false
}

// spellsAWord is the well-formedness test for scripts with no orthographic rule
// of their own: letters, and the marks that occur inside words.
func spellsAWord(text string) bool {
	letters := false
	for _, r := range text {
		switch {
		case unicode.IsLetter(r):
			letters = true
		case r == ' ' || r == '-' || r == '\'' || r == '.' || unicode.IsDigit(r):
		default:
			return false
		}
	}
	return letters
}

func isThai(text string) bool {
	for _, r := range text {
		if isThaiRune(r) {
			return true
		}
	}
	return false
}

func isThaiRune(r rune) bool { return r >= 0x0E00 && r <= 0x0E7F }
