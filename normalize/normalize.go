// Package normalize canonicalizes Thai text so that strings which read the same
// to a human become the same bytes. It is a faithful port of PyThaiNLP's
// pythainlp.util.normalize (rule-based; it does NOT apply Unicode NFC — apply
// golang.org/x/text/unicode/norm separately if you need that).
//
// Normalize runs, in order:
//
//	RemoveZW               strip zero-width space / non-joiner
//	RemoveDupSpaces        collapse repeated spaces and blank lines
//	RemoveSpacesBeforeMarks drop a stray space before a combining mark
//	RemoveRepeatVowels     reorder vowels/tone marks, drop repeats
//	RemoveDangling         drop combining marks with no base (start / after space)
//
// Normalize is deterministic and faithful to the PyThaiNLP output. (Like the
// original it is not strictly idempotent for text ending in a dangling mark
// after a space — a quirk kept for exact parity.)
package normalize

import (
	"regexp"
	"strings"
)

// Thai character classes (codepoints match PyThaiNLP).
var (
	tonemarks = "่้๊๋"    // ่ ้ ๊ ๋
	aboveV    = "ัิีึืํ็" // ั ิ ี ึ ื ํ ็
	belowV    = "ุู"      // ุ ู
	followV   = "ะาำๅ"    // ะ า ำ ๅ
	leadV     = "เแโใไ"   // เ แ โ ใ ไ
	danglings = aboveV + belowV + tonemarks + "ฺ์ํ๎"
	noRepeat  = followV + leadV + aboveV + belowV + "ฺ์ํ๎"
	// Consonants ก..ฮ EXCLUDING ฤ (U+0E24) and ฦ (U+0E26), which are vowel-like.
	thaiConsonants = rangeExcept(0x0e01, 0x0e2e, 0x0e24, 0x0e26)
	thaiVowels     = "ฤฦ" + followV + leadV + aboveV + belowV
)

func rangeExcept(lo, hi rune, except ...rune) string {
	skip := map[rune]bool{}
	for _, e := range except {
		skip[e] = true
	}
	var b strings.Builder
	for r := lo; r <= hi; r++ {
		if !skip[r] {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// Compiled patterns (built once).
var (
	reNewlines = regexp.MustCompile(`[ \n]*\n[ \n]*`)
	reToneRun  = regexp.MustCompile("[" + tonemarks + "]+")

	// reorder pairs (applied once each, in order)
	reOrder2 = regexp.MustCompile("([" + tonemarks + "์]+)([" + aboveV + belowV + "]+)")
	reOrder3 = regexp.MustCompile("ํ([" + tonemarks + "]*)า")
	reOrder4 = regexp.MustCompile("([" + followV + "]+)([" + tonemarks + "]+)")
	reOrder5 = regexp.MustCompile("([^ฤฦ])ๅ")

	// dangling removal
	reDanglingStart = regexp.MustCompile("^[" + danglings + "]+")
	reDanglingSpace = regexp.MustCompile(" +[" + danglings + "]+")

	// no-repeat: one per char in noRepeat, "(ch *)+ch" -> "ch"
	noRepeatRules = buildNoRepeat()
)

type reRule struct {
	re *regexp.Regexp
	ch string
}

func buildNoRepeat() []reRule {
	var rules []reRule
	for _, ch := range noRepeat {
		c := regexp.QuoteMeta(string(ch))
		rules = append(rules, reRule{regexp.MustCompile("(" + c + "[ ]*)+" + c), string(ch)})
	}
	return rules
}

// Normalize returns the canonical form of text (see package doc for the rules).
func Normalize(text string) string {
	if !needsNormalize(text) {
		return text // fast path: nothing any rule could change
	}
	text = RemoveZW(text)
	text = RemoveDupSpaces(text)
	text = RemoveSpacesBeforeMarks(text)
	text = RemoveRepeatVowels(text)
	text = RemoveDangling(text)
	return text
}

// needsNormalize reports whether text contains anything a normalization rule
// could act on. It returns false only when no rule can possibly change the text,
// letting Normalize return the input untouched with zero allocation. Triggers:
// any Thai vowel/tone/sign (U+0E30–U+0E4E), a zero-width char, a newline/tab, or
// a leading/trailing/doubled space. Consonants and digits alone never trigger a
// change, so consonant-only or non-Thai text takes the fast path.
func needsNormalize(text string) bool {
	prevSpace := false
	lastSpace := false
	for i, r := range text {
		if r >= 0x0e30 && r <= 0x0e4e {
			return true
		}
		switch r {
		case '​', '‌', '\n', '\t':
			return true
		case ' ':
			if i == 0 || prevSpace {
				return true // leading or doubled space
			}
			prevSpace, lastSpace = true, true
			continue
		}
		prevSpace, lastSpace = false, false
	}
	return lastSpace // trailing space
}

// RemoveZW strips zero-width space (U+200B) and zero-width non-joiner (U+200C).
func RemoveZW(text string) string {
	text = strings.ReplaceAll(text, "​", "")
	text = strings.ReplaceAll(text, "‌", "")
	return text
}

// RemoveDupSpaces collapses repeated spaces to one and blank/space-padded
// newlines to a single newline, then trims surrounding whitespace.
func RemoveDupSpaces(text string) string {
	for strings.Contains(text, "  ") {
		text = strings.ReplaceAll(text, "  ", " ")
	}
	text = reNewlines.ReplaceAllString(text, "\n")
	return strings.TrimSpace(text)
}

// RemoveSpacesBeforeMarks removes a single space that sits between a consonant
// and a following combining mark, unless the consonant is itself preceded by a
// vowel (a conservative rule that fixes stray spaces without breaking words).
// Ports the PyThaiNLP negative-lookbehind regex with a single left-to-right pass.
func RemoveSpacesBeforeMarks(text string) string {
	rs := []rune(text)
	out := make([]rune, 0, len(rs))
	for i := 0; i < len(rs); i++ {
		if i+2 < len(rs) &&
			strings.ContainsRune(thaiConsonants, rs[i]) &&
			rs[i+1] == ' ' &&
			strings.ContainsRune(danglings, rs[i+2]) &&
			!(i >= 1 && strings.ContainsRune(thaiVowels, rs[i-1])) {
			out = append(out, rs[i], rs[i+2]) // drop the space
			i += 2
			continue
		}
		out = append(out, rs[i])
	}
	return string(out)
}

// RemoveRepeatVowels reorders tone marks/vowels to canonical order, then removes
// repeated vowels/signs and collapses repeated tone marks to the last one.
func RemoveRepeatVowels(text string) string {
	text = reorderVowels(text)
	for _, r := range noRepeatRules {
		text = r.re.ReplaceAllString(text, r.ch)
	}
	// collapse a run of tone marks to its last mark
	text = reToneRun.ReplaceAllStringFunc(text, func(m string) string {
		r := []rune(m)
		return string(r[len(r)-1])
	})
	return text
}

func reorderVowels(text string) string {
	text = strings.ReplaceAll(text, "เเ", "แ")         // Sara E + Sara E -> Sara Ae
	text = reOrder2.ReplaceAllString(text, "${2}${1}") // TONE + ABV/BLW vowel -> vowel + TONE
	text = reOrder3.ReplaceAllString(text, "${1}ำ")    // Nikhahit + TONE* + Sara Aa -> TONE* + Sara Am
	text = reOrder4.ReplaceAllString(text, "${2}${1}") // FOLLOW vowel + TONE -> TONE + FOLLOW vowel
	text = reOrder5.ReplaceAllString(text, "${1}า")    // Lakkhangyao -> Sara Aa (except after ฤ/ฦ)
	return text
}

// RemoveDangling removes combining marks that have no base character: at the
// start of the text, or immediately after a space.
func RemoveDangling(text string) string {
	text = reDanglingStart.ReplaceAllString(text, "")
	text = reDanglingSpace.ReplaceAllString(text, " ")
	return text
}
