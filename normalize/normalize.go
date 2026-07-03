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
	reToneRun = regexp.MustCompile("[" + tonemarks + "]+")

	// reorder pairs (applied once each, in order)
	reOrder2 = regexp.MustCompile("([" + tonemarks + "์]+)([" + aboveV + belowV + "]+)")
	reOrder3 = regexp.MustCompile("ํ([" + tonemarks + "]*)า")
	reOrder4 = regexp.MustCompile("([" + followV + "]+)([" + tonemarks + "]+)")
	reOrder5 = regexp.MustCompile("([^ฤฦ])ๅ")

	// dangling removal
	reDanglingStart = regexp.MustCompile("^[" + danglings + "]+")
	reDanglingSpace = regexp.MustCompile(" +[" + danglings + "]+")

	noRepeatSet = buildSet(noRepeat)
)

func buildSet(s string) map[rune]bool {
	m := make(map[rune]bool, len(s))
	for _, r := range s {
		m[r] = true
	}
	return m
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

// RemoveDupSpaces collapses each run of spaces/newlines to a single character
// (a newline if the run contained one, else a space), then trims surrounding
// whitespace. Single O(n) pass.
func RemoveDupSpaces(text string) string {
	rs := []rune(text)
	out := make([]rune, 0, len(rs))
	for i := 0; i < len(rs); {
		if rs[i] == ' ' || rs[i] == '\n' {
			hasNL := false
			j := i
			for j < len(rs) && (rs[j] == ' ' || rs[j] == '\n') {
				if rs[j] == '\n' {
					hasNL = true
				}
				j++
			}
			if hasNL {
				out = append(out, '\n')
			} else {
				out = append(out, ' ')
			}
			i = j
			continue
		}
		out = append(out, rs[i])
		i++
	}
	return strings.TrimSpace(string(out))
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
	text = collapseRepeats(text)
	// collapse a run of tone marks to its last mark (only if any tone mark present)
	if strings.ContainsAny(text, tonemarks) {
		text = reToneRun.ReplaceAllStringFunc(text, func(m string) string {
			r := []rune(m)
			return string(r[len(r)-1])
		})
	}
	return text
}

// collapseRepeats replaces a repeated vowel/sign — the same character repeated,
// optionally separated by spaces — with a single one, dropping the intervening
// characters and spaces. It ports the per-character PyThaiNLP rules
// "(ch[ ]*)+ch" -> "ch" as one left-to-right pass (tone marks are handled
// separately). Trailing spaces after the last repeat are preserved.
func collapseRepeats(text string) string {
	rs := []rune(text)
	out := make([]rune, 0, len(rs))
	for i := 0; i < len(rs); i++ {
		r := rs[i]
		if noRepeatSet[r] {
			last := i
			for k := i + 1; k < len(rs); {
				s := k
				for s < len(rs) && rs[s] == ' ' {
					s++
				}
				if s < len(rs) && rs[s] == r {
					last = s
					k = s + 1
				} else {
					break
				}
			}
			out = append(out, r)
			i = last // skip the collapsed repeats (and spaces between them)
			continue
		}
		out = append(out, r)
	}
	return string(out)
}

// reorderVowels applies the reorder rules, each guarded by a cheap check so its
// regex only runs when the triggering characters are actually present.
func reorderVowels(text string) string {
	if strings.Contains(text, "เเ") { // Sara E + Sara E -> Sara Ae
		text = strings.ReplaceAll(text, "เเ", "แ")
	}
	if strings.ContainsAny(text, tonemarks+"์") && strings.ContainsAny(text, aboveV+belowV) {
		text = reOrder2.ReplaceAllString(text, "${2}${1}") // TONE + ABV/BLW vowel -> vowel + TONE
	}
	if strings.ContainsRune(text, 'ํ') {
		text = reOrder3.ReplaceAllString(text, "${1}ำ") // Nikhahit + TONE* + Sara Aa -> TONE* + Sara Am
	}
	if strings.ContainsAny(text, followV) && strings.ContainsAny(text, tonemarks) {
		text = reOrder4.ReplaceAllString(text, "${2}${1}") // FOLLOW vowel + TONE -> TONE + FOLLOW vowel
	}
	if strings.ContainsRune(text, 'ๅ') {
		text = reOrder5.ReplaceAllString(text, "${1}า") // Lakkhangyao -> Sara Aa (except after ฤ/ฦ)
	}
	return text
}

// thaiCCC returns the Unicode canonical combining class of a Thai (or the two
// Patani-Malay generic) combining marks, or 0 for base characters and class-0
// marks. Class 0 marks form an opaque boundary and are never reordered.
// Values per UTC L2/18-216 / UCD: phinthu=9, sara u/uu=103, tone marks=107,
// macron-below=220, tilde=230.
func thaiCCC(r rune) int {
	switch r {
	case 0x0E3A: // phinthu (nukta)
		return 9
	case 0x0E38, 0x0E39: // sara u, sara uu (below vowels)
		return 103
	case 0x0E48, 0x0E49, 0x0E4A, 0x0E4B: // mai ek/tho/tri/chattawa (tone marks)
		return 107
	case 0x0331: // combining macron below (generic, Patani Malay)
		return 220
	case 0x0303: // combining tilde (generic, Patani Malay)
		return 230
	}
	return 0
}

// Reorder puts combining marks into Unicode canonical order: within each run of
// consecutive non-zero-class marks it stable-sorts by combining class. This is
// exactly what Unicode NFC does for Thai (which has no canonical composition),
// so two sequences that render identically but were typed in different mark
// order (e.g. tone-before-vowel vs vowel-before-tone) become identical bytes —
// something PyThaiNLP's normalize() does not do. Class-0 marks are opaque
// boundaries and are never moved (matching Unicode). Verified against
// golang.org/x/text/unicode/norm.NFC in the test suite.
func Reorder(text string) string {
	rs := []rune(text)
	changed := false
	for i := 0; i < len(rs); {
		if thaiCCC(rs[i]) == 0 {
			i++
			continue
		}
		j := i
		for j < len(rs) && thaiCCC(rs[j]) > 0 {
			j++
		}
		// stable insertion sort rs[i:j] ascending by combining class
		for a := i + 1; a < j; a++ {
			for b := a; b > i && thaiCCC(rs[b-1]) > thaiCCC(rs[b]); b-- {
				rs[b-1], rs[b] = rs[b], rs[b-1]
				changed = true
			}
		}
		i = j
	}
	if !changed {
		return text
	}
	return string(rs)
}

// Canonical is the strongest canonicalization: PyThaiNLP-style Normalize plus
// Unicode canonical mark reordering (Reorder). Use it for exact-match / dedup /
// hashing where two visually-identical strings must have identical bytes. Not
// byte-identical to PyThaiNLP (which omits Unicode normalization) — that is the
// point.
func Canonical(text string) string {
	return Reorder(Normalize(text))
}

// RemoveDangling removes combining marks that have no base character: at the
// start of the text, or immediately after a space.
func RemoveDangling(text string) string {
	// remove dangling marks at the very start
	if r := firstRune(text); r != 0 && strings.ContainsRune(danglings, r) {
		text = reDanglingStart.ReplaceAllString(text, "")
	}
	// remove dangling marks right after a space
	if strings.Contains(text, " ") {
		text = reDanglingSpace.ReplaceAllString(text, " ")
	}
	return text
}

func firstRune(s string) rune {
	for _, r := range s {
		return r
	}
	return 0
}
