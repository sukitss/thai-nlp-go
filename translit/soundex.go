package translit

// Thai soundex variants ported from PyThaiNLP: Udom83 (Udompanich 1983) and
// LK82 (Lorchirachoonkul 1982). Together with Metasound (see translit.go) they
// give the classic Thai phonetic-key trio. Each maps a word to a fixed-length
// code so that similar-sounding spellings share a key. Verified byte-for-byte
// against PyThaiNLP in the test suite.

import (
	"regexp"
	"strings"
)

func buildTrans(from, to string) map[rune]rune {
	fr, tr := []rune(from), []rune(to)
	m := make(map[rune]rune, len(fr))
	for i, r := range fr {
		m[r] = tr[i]
	}
	return m
}

func translate(s string, m map[rune]rune) string {
	return strings.Map(func(r rune) rune {
		if v, ok := m[r]; ok {
			return v
		}
		return r
	}, s)
}

func firstRunes(s string, n int) string {
	rs := []rune(s)
	if len(rs) > n {
		rs = rs[:n]
	}
	return string(rs)
}

func mapRune(r rune, m map[rune]rune) rune {
	if v, ok := m[r]; ok {
		return v
	}
	return r
}

// hasThaiMark reports whether text contains any Thai vowel/tone/sign (U+0E30–4C).
func hasThaiMark(text string) bool {
	for _, r := range text {
		if r >= 0x0e30 && r <= 0x0e4c {
			return true
		}
	}
	return false
}

// ---------- Udom83 ----------

var (
	udT1 = buildTrans(udTrans1From, udTrans1To)
	udT2 = buildTrans(udTrans2From, udTrans2To)

	udRE1  = regexp.MustCompile(`รร([\x{0e40}-\x{0e44}])`)
	udRE2  = regexp.MustCompile(`รร([` + thaiCons + `][` + thaiCons + `\x{0e40}-\x{0e44}])`)
	udRE3  = regexp.MustCompile(`รร([` + thaiCons + `][\x{0e30}-\x{0e39}\x{0e48}-\x{0e4c}])`)
	udRE4  = regexp.MustCompile(`รร`)
	udRE5  = regexp.MustCompile(`ไ([` + thaiCons + `]ย)`)
	udRE6  = regexp.MustCompile(`[ไใ]([` + thaiCons + `])`)
	udRE7  = regexp.MustCompile(`\x{0e33}(ม[\x{0e30}-\x{0e39}])`)
	udRE8  = regexp.MustCompile(`\x{0e33}ม`)
	udRE9  = regexp.MustCompile(`\x{0e33}`)
	udRE10 = regexp.MustCompile(`จน์|มณ์|ณฑ์|ทร์|ตร์|[` + thaiCons + `]\x{0e4c}|[` + thaiCons + `][\x{0e30}-\x{0e39}]\x{0e4c}`)
	udRE11 = regexp.MustCompile(`[\x{0e30}-\x{0e4c}]`)
)

// Udom83 returns the Udom83 Thai soundex of text (7-char code). Each rewrite
// regex is guarded by a cheap Contains check so it only runs when its trigger
// characters are present (most words trigger only a couple).
func Udom83(text string) string {
	if text == "" {
		return ""
	}
	if strings.Contains(text, "รร") {
		text = udRE1.ReplaceAllString(text, "ัน${1}")
		text = udRE2.ReplaceAllString(text, "ั${1}")
		text = udRE3.ReplaceAllString(text, "ัน${1}")
		text = udRE4.ReplaceAllString(text, "ัน")
	}
	if strings.ContainsRune(text, 'ไ') {
		text = udRE5.ReplaceAllString(text, "${1}")
	}
	if strings.ContainsAny(text, "ไใ") {
		text = udRE6.ReplaceAllString(text, "${1}ย")
	}
	if strings.ContainsRune(text, 'ำ') {
		text = udRE7.ReplaceAllString(text, "ม${1}")
		text = udRE8.ReplaceAllString(text, "ม")
		text = udRE9.ReplaceAllString(text, "ม")
	}
	if strings.ContainsRune(text, '์') {
		text = udRE10.ReplaceAllString(text, "")
	}
	if hasThaiMark(text) {
		text = udRE11.ReplaceAllString(text, "")
	}
	if text == "" {
		return ""
	}
	rs := []rune(text)
	sd := translate(string(rs[0]), udT1) + translate(string(rs[1:]), udT2) + "000000"
	return firstRunes(sd, 7)
}

// ---------- LK82 ----------

var (
	lkT1 = buildTrans(lkTrans1From, lkTrans1To)
	lkT2 = buildTrans(lkTrans2From, lkTrans2To)

	lkKarant = regexp.MustCompile(`จน์|มณ์|ณฑ์|ทร์|ตร์|[\x{0e01}-\x{0e2e}]\x{0e4c}|[\x{0e01}-\x{0e2e}][\x{0e30}-\x{0e39}]\x{0e4c}`)
	lkSign   = regexp.MustCompile(`[\x{0e2f}\x{0e3a}\x{0e46}\x{0e47}\x{0e4d}]`)
	lkTone   = regexp.MustCompile(`[\x{0e48}-\x{0e4b}]`)
)

// sentinel rune for a "separator" (Python's "" append): kept in the code
// sequence so it breaks de-duplication, but omitted from the final string.
const lkSep = rune(0)

// LK82 returns the LK82 Thai soundex of text (5-char code). Rune-based (no
// per-character string allocation) with the cleanup regexes guarded.
func LK82(text string) string {
	if text == "" {
		return ""
	}
	if strings.ContainsAny(text, "่้๊๋") { // tone marks
		text = lkTone.ReplaceAllString(text, "")
	}
	if strings.ContainsRune(text, '์') {
		text = lkKarant.ReplaceAllString(text, "")
	}
	if strings.ContainsAny(text, "ฯฺๆ็ํ") { // 0e2f,0e3a,0e46,0e47,0e4d
		text = lkSign.ReplaceAllString(text, "")
	}
	if text == "" {
		return ""
	}
	rs := []rune(text)
	res := make([]rune, 0, len(rs)+1)
	if rs[0] >= 'ก' && rs[0] <= 'ฮ' {
		res = append(res, mapRune(rs[0], lkT1))
		rs = rs[1:]
	} else {
		if len(rs) > 1 {
			res = append(res, mapRune(rs[1], lkT1))
		}
		res = append(res, mapRune(rs[0], lkT2))
		rs = rs[2:]
	}

	iv := -2 // sentinel (Python None): never equals i-1 for i>=0
	n := len(rs)
	below := func(r rune) bool { return r == 'ึ' || r == 'ื' || r == 'ุ' || r == 'ู' }
	for i, c := range rs {
		switch {
		case c == 'ะ' || c == 'ั' || c == 'ิ' || c == 'ี': // separator only
			iv = i
			res = append(res, lkSep)
		case c == 'า' || c == 'ึ' || c == 'ื' || c == 'ู' || c == 'ๅ':
			iv = i
			res = append(res, mapRune(c, lkT2))
		case c == 'ุ':
			iv = i
			if i == 0 || (rs[i-1] != 'ต' && rs[i-1] != 'ธ') {
				res = append(res, mapRune(c, lkT2))
			} else {
				res = append(res, lkSep)
			}
		case c == 'ห' || c == 'อ':
			if i+1 < n && below(rs[i+1]) {
				res = append(res, mapRune(c, lkT2))
			}
		case c == 'ย' || c == 'ร' || c == 'ฤ' || c == 'ฦ' || c == 'ว':
			if iv == i-1 || (i+1 < n && below(rs[i+1])) {
				res = append(res, mapRune(c, lkT2))
			}
		default:
			res = append(res, mapRune(c, lkT2))
		}
	}

	if len(res) == 0 {
		return "00000"
	}
	// de-duplicate consecutive equal codes (compare each to the immediately
	// preceding element; separators break runs), emit skipping separators, pad 5.
	var b strings.Builder
	if res[0] != lkSep {
		b.WriteRune(res[0])
	}
	for i := 1; i < len(res); i++ {
		if res[i] != res[i-1] && res[i] != lkSep {
			b.WriteRune(res[i])
		}
	}
	return firstRunes(b.String()+"0000", 5)
}
