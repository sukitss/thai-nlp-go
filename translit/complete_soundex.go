package translit

// Complete Soundex for Thai words, ported from PyThaiNLP's CompleteSoundex
// (pythainlp/soundex/complete_soundex.py).
//
// Original algorithm:
//
//	Chalermpol Tapsai, Phayung Meesad, and Choochart Haruechaiyasak. 2020.
//	Complete Soundex for Thai Words Similarity Analysis.
//	Information Technology Journal KMUTNB. 2020 June 30;16(1):46-59.
//	https://ph01.tci-thaijo.org/index.php/IT_Journal/article/view/241562
//
// This is a faithful port of the per-syllable ENCODING. PyThaiNLP's
// complete_soundex() additionally runs a CRF syllable tokenizer
// (python-crfsuite) to split multi-syllable words before encoding each
// syllable. That CRF model is out of scope here, so:
//
//   - CompleteSoundex encodes assuming a SINGLE syllable. It matches
//     PyThaiNLP's complete_soundex() byte-for-byte for single-syllable input
//     (verified in complete_soundex_test.go against the golden set).
//   - CompleteSoundexSyllables takes already-split syllables (e.g. from an
//     external syllable tokenizer) and joins their codes exactly as PyThaiNLP
//     does for multi-syllable words.
//   - CompleteSoundexSimilarity is the character-wise similarity from the paper.
//
// The maps live in complete_soundex_tables.go (auto-generated from PyThaiNLP).

import (
	"regexp"
	"strings"
)

// clean_text: remove silent characters (karan/thanthakhat) — a consonant,
// optional lower/upper vowel, then thanthakhat (U+0E4C).
var csKaranRE = regexp.MustCompile(`[\x{0e01}-\x{0e2e}][\x{0e30}-\x{0e39}]?\x{0e4c}`)

// leadingVowels are the pre-posed vowels handled before the initial consonant.
const csLeadingVowels = "เแโไใ" // เ แ โ ไ ใ

// consonant range ก(0e01)..ฮ(0e2e)
func csIsConsonant(r rune) bool { return r >= 0x0e01 && r <= 0x0e2e }

func csClean(text string) string { return csKaranRE.ReplaceAllString(text, "") }

func csInSet(r rune, set string) bool { return strings.ContainsRune(set, r) }

// csSyl is a syllable plus an optional implicit-vowel rule ("a", "o", or "").
type csSyl struct {
	text string
	rule string
}

// heuristicSplit mirrors CompleteSoundex.heuristic_split.
func csHeuristicSplit(text string) []csSyl {
	rs := []rune(text)

	// 0. อัต pattern (e.g. อัตรา -> อัต / ตรา), keep ต with the second part.
	if strings.HasPrefix(text, "อัต") && len(rs) > 3 { // อัต
		return []csSyl{{"อัต", ""}, {"ต" + string(rs[3:]), ""}}
	}

	// 1. Aksorn Nam with Ro Han (e.g. สวรรค์ -> ส / วรรค์).
	if len(rs) >= 4 &&
		csInSet(rs[0], "ขฃฉฐถผฝศษสฮกจดตฎฏบปอ") &&
		rs[1] == 'ว' && rs[2] == 'ร' && rs[3] == 'ร' { // ว ร ร
		return []csSyl{{string(rs[0]), "a"}, {string(rs[1:]), ""}}
	}

	// 2. Two consonants without vowel (e.g. กม -> ก-a ม-a).
	if len(rs) == 2 && csIsConsonant(rs[0]) && csIsConsonant(rs[1]) {
		return []csSyl{{string(rs[0]), "a"}, {string(rs[1]), "a"}}
	}

	// 3. Three consonants -> C1-a C2C3-o (e.g. กมล).
	if len(rs) == 3 && csIsConsonant(rs[0]) && csIsConsonant(rs[1]) && csIsConsonant(rs[2]) {
		return []csSyl{{string(rs[0]), "a"}, {string(rs[1:]), "o"}}
	}

	// 4. Three consonants + vowel -> C1-a C2-a C3-V (e.g. กมลา).
	if len(rs) == 4 && csIsConsonant(rs[0]) && csIsConsonant(rs[1]) && csIsConsonant(rs[2]) &&
		rs[3] >= 0x0e32 && rs[3] <= 0x0e39 { // า..ู
		return []csSyl{{string(rs[0]), "a"}, {string(rs[1]), "a"}, {string(rs[2:]), ""}}
	}

	return []csSyl{{text, ""}}
}

// csDetectCluster mirrors CompleteSoundex._detect_cluster.
func csDetectCluster(chars []rune, idx int, leadingVowel rune) bool {
	if idx+1 < len(chars) {
		nc := chars[idx+1]
		// combining vowel marks or tone marks
		if csInSet(nc, "ะัิีึืุู") { // ะ ั ิ ี ึ ื ุ ู
			return true
		}
		if _, ok := csToneMap[string(nc)]; ok {
			return true
		}
		if leadingVowel != 0 &&
			nc != 'ร' && nc != 'ล' && nc != 'ว' && // ร ล ว
			!csInSet(nc, csThaiConsonants) &&
			nc != 'า' { // า
			return true
		}
		return false
	}
	if leadingVowel != 0 {
		return true
	}
	return false
}

// csProcessVowelChar mirrors CompleteSoundex._process_vowel_char.
func csProcessVowelChar(c rune, leadingVowel rune, vowelCode, finalCode string) (string, string) {
	switch {
	case leadingVowel == 'เ' && c == 'ื': // เ + ื -> uea
		vowelCode = "BV"
	case leadingVowel == 'เ' && c == 'อ': // เ + อ -> oe
		if vowelCode != "BV" {
			vowelCode = "9R"
		}
	case c == 'ำ': // ำ
		vowelCode = "1A"
		finalCode = "ม" // ม
	case c == 'อ' && leadingVowel == 0 && vowelCode == "": // อ as vowel
		vowelCode = "8P"
	default:
		if v, ok := csVowelMap[string(c)]; ok && v != "" {
			vowelCode = v
		}
	}

	// 'ะ' shortening
	if c == 'ะ' { // ะ
		switch vowelCode {
		case "5J":
			vowelCode = "5I"
		case "6L":
			vowelCode = "6K"
		case "7N":
			vowelCode = "7M"
		case "1B":
			vowelCode = "1A"
		}
	}

	return vowelCode, finalCode
}

// csProcessSyllable mirrors CompleteSoundex.process_syllable.
func csProcessSyllable(syl, implicitRule string) string {
	chars := []rune(syl)
	idx := 0

	// A. Leading vowel.
	var leadingVowel rune
	if idx < len(chars) && csInSet(chars[idx], csLeadingVowels) {
		leadingVowel = chars[idx]
		idx++
	}

	// B. Initial consonant and cluster.
	var initChar rune
	initCode := ""
	clusterChar := "-"
	if idx < len(chars) {
		initChar = chars[idx]
		if initChar == 'ท' && idx+1 < len(chars) && chars[idx+1] == 'ร' { // ทร -> ซ
			initCode = "ซซ" // ซซ
			idx += 2
			clusterChar = "-"
		} else {
			if v, ok := csInitialMap[string(initChar)]; ok {
				initCode = v
			} else {
				initCode = "xx"
			}
			idx++
			if idx < len(chars) && (chars[idx] == 'ร' || chars[idx] == 'ล' || chars[idx] == 'ว') { // ร ล ว
				if csDetectCluster(chars, idx, leadingVowel) {
					clusterChar = string(chars[idx])
					idx++
				}
			}
		}
	}

	// D. Map leading vowel to code.
	vowelCode := ""
	finalCode := "-"
	switch leadingVowel {
	case 'โ': // โ
		vowelCode = "7N"
	case 'ไ': // ไ
		vowelCode = "1A"
		finalCode = "ย" // ย
	case 'ใ': // ใ
		vowelCode = "1A"
		finalCode = "ย" // ย
	case 'แ': // แ
		vowelCode = "6L"
	case 'เ': // เ
		vowelCode = "5J"
	}

	// E. Scan remaining for vowels, tones, finals.
	toneCode := "0"
	var finalCandidates []rune
	for _, c := range chars[idx:] {
		if t, ok := csToneMap[string(c)]; ok {
			toneCode = t
		} else if csInSet(c, "ะัาิีึืุู") || // ะ ั า ิ ี ึ ื ุ ู
			(c == 'อ' && leadingVowel == 'เ') || // อ after เ
			c == 'ำ' { // ำ
			vowelCode, finalCode = csProcessVowelChar(c, leadingVowel, vowelCode, finalCode)
		} else {
			finalCandidates = append(finalCandidates, c)
		}
	}

	// F. Final consonant processing.
	droppedR := false
	if finalCode == "-" {
		if strings.Contains(syl, "รร") { // รร
			vowelCode = "1A"
			if len(finalCandidates) > 0 {
				f := finalCandidates[len(finalCandidates)-1]
				if v, ok := csFinalMap[string(f)]; ok {
					finalCode = v
				} else {
					finalCode = "-"
				}
			} else {
				finalCode = "น" // น
			}
		} else if len(finalCandidates) > 0 {
			raw := finalCandidates
			var f rune
			if len(raw) >= 2 {
				_, penultIsFinal := csFinalMap[string(raw[len(raw)-1])]
				if raw[len(raw)-2] == 'ร' && penultIsFinal { // ร + valid final
					f = raw[len(raw)-1]
					droppedR = true
				} else if raw[len(raw)-2] == 'ต' && raw[len(raw)-1] == 'ร' { // ...ตร
					f = 'ต' // ต
					droppedR = true
				} else {
					f = raw[len(raw)-1]
				}
			} else {
				f = raw[len(raw)-1]
			}
			if v, ok := csFinalMap[string(f)]; ok {
				finalCode = v
			} else {
				finalCode = "-"
			}
		}
	}

	// Special format (tone before final).
	specialFormat := csCheckSpecialFormat(initChar, finalCandidates, vowelCode)

	// G. Implicit vowel / defaults.
	if vowelCode == "" {
		switch implicitRule {
		case "a":
			vowelCode = "1A"
		case "o":
			vowelCode = "7M"
		default:
			vowelCode = "7M"
		}
	}

	// Specific fixes.
	if leadingVowel == 'โ' { // โ
		vowelCode = "7N"
	}
	if leadingVowel == 'แ' { // แ
		vowelCode = "6L"
	}

	// H. so-sua (ส) adjustment.
	if initChar == 'ส' && initCode == "ซศ" { // ส, code ซศ
		if implicitRule == "" {
			sr := []rune(syl)
			if len(sr) >= 2 {
				allRoLoWo := true
				hasConsonant := false
				for _, c := range sr[1:] {
					if csIsConsonant(c) {
						hasConsonant = true
						if c != 'ร' && c != 'ล' && c != 'ว' { // ร ล ว
							allRoLoWo = false
						}
					}
				}
				if !hasConsonant || allRoLoWo {
					initCode = "ซซ" // ซซ
				}
			}
		}
	}

	// I. Format output.
	if specialFormat {
		return initCode + vowelCode + toneCode + finalCode + clusterChar
	}
	if droppedR && (finalCode == "ก" || finalCode == "-") { // ก or none
		return initCode + vowelCode + "-" + finalCode + toneCode + clusterChar
	}
	return initCode + vowelCode + finalCode + toneCode + clusterChar
}

// csCheckSpecialFormat mirrors CompleteSoundex._check_special_format.
func csCheckSpecialFormat(initChar rune, finalCandidates []rune, vowelCode string) bool {
	if initChar == 'ญ' || initChar == 'ย' || initChar == 'น' { // ญ ย น
		return true
	}
	for _, c := range finalCandidates {
		if c == 'ญ' || c == 'ณ' { // ญ ณ
			return true
		}
	}
	if vowelCode == "1A" {
		for _, c := range finalCandidates {
			if c == 'น' { // น
				return true
			}
		}
	}
	return false
}

// csAddAsterisk appends the trailing "*" marker used by PyThaiNLP for words
// involving ญ/ณ+ย patterns. startsWithYoYing reports whether any syllable
// (or the text, for single-syllable) starts with ญ.
func csAddAsterisk(result, text string, startsWithYoYing bool) string {
	if strings.Contains(text, "ญญ") || // ญญ
		(strings.Contains(text, "ญ") && strings.Contains(text, "ย")) || // ญ & ย
		(strings.Contains(text, "ณ") && strings.Contains(text, "ย")) || // ณ & ย
		startsWithYoYing {
		return result + "*"
	}
	return result
}

// CompleteSoundex encodes a Thai word into its Complete Soundex code, assuming
// the input is a SINGLE syllable. This matches PyThaiNLP's complete_soundex()
// byte-for-byte for single-syllable words (multi-syllable words require a
// syllable tokenizer — see CompleteSoundexSyllables). Empty input returns "".
func CompleteSoundex(text string) string {
	text = csClean(text)
	if text == "" {
		return ""
	}
	var b strings.Builder
	for _, s := range csHeuristicSplit(text) {
		b.WriteString(csProcessSyllable(s.text, s.rule))
	}
	return csAddAsterisk(b.String(), text, strings.HasPrefix(text, "ญ")) // ญ
}

// CompleteSoundexSyllables encodes each pre-split syllable and joins the codes
// exactly as PyThaiNLP's complete_soundex() does for multi-syllable words. Pass
// syllables produced by a syllable tokenizer (already cleaned of karan). This
// yields the full multi-syllable code for callers that already have syllables.
// Empty input returns "".
func CompleteSoundexSyllables(sylls []string) string {
	if len(sylls) == 0 {
		return ""
	}
	var b strings.Builder
	startsWithYoYing := false
	for _, syl := range sylls {
		if strings.HasPrefix(syl, "ญ") { // ญ
			startsWithYoYing = true
		}
		for _, s := range csHeuristicSplit(syl) {
			b.WriteString(csProcessSyllable(s.text, s.rule))
		}
	}
	text := strings.Join(sylls, "")
	return csAddAsterisk(b.String(), text, startsWithYoYing)
}

// CompleteSoundexSimilarity computes the character-wise similarity between two
// Complete Soundex codes, per Tapsai et al. (2020) Eq. (1): the number of
// position-wise matching characters divided by max(len1, len2). Two empty codes
// score 1.0; one empty scores 0.0. Length and comparison are by Unicode code
// point, matching PyThaiNLP.
func CompleteSoundexSimilarity(code1, code2 string) float64 {
	if code1 == "" && code2 == "" {
		return 1.0
	}
	if code1 == "" || code2 == "" {
		return 0.0
	}
	r1, r2 := []rune(code1), []rune(code2)
	maxLen := len(r1)
	if len(r2) > maxLen {
		maxLen = len(r2)
	}
	minLen := len(r1)
	if len(r2) < minLen {
		minLen = len(r2)
	}
	matches := 0
	for i := 0; i < minLen; i++ {
		if r1[i] == r2[i] {
			matches++
		}
	}
	return float64(matches) / float64(maxLen)
}
