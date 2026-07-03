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

// Udom83 returns the Udom83 Thai soundex of text (7-char code).
func Udom83(text string) string {
	if text == "" {
		return ""
	}
	text = udRE1.ReplaceAllString(text, "ัน${1}")
	text = udRE2.ReplaceAllString(text, "ั${1}")
	text = udRE3.ReplaceAllString(text, "ัน${1}")
	text = udRE4.ReplaceAllString(text, "ัน")
	text = udRE5.ReplaceAllString(text, "${1}")
	text = udRE6.ReplaceAllString(text, "${1}ย")
	text = udRE7.ReplaceAllString(text, "ม${1}")
	text = udRE8.ReplaceAllString(text, "ม")
	text = udRE9.ReplaceAllString(text, "ม")
	text = udRE10.ReplaceAllString(text, "")
	text = udRE11.ReplaceAllString(text, "")
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

// LK82 returns the LK82 Thai soundex of text (5-char code).
func LK82(text string) string {
	if text == "" {
		return ""
	}
	text = lkTone.ReplaceAllString(text, "") // remove tone marks
	text = lkKarant.ReplaceAllString(text, "")
	text = lkSign.ReplaceAllString(text, "")
	if text == "" {
		return ""
	}
	rs := []rune(text)
	var res []string
	if rs[0] >= 'ก' && rs[0] <= 'ฮ' {
		res = append(res, translate(string(rs[0]), lkT1))
		rs = rs[1:]
	} else {
		if len(rs) > 1 {
			res = append(res, translate(string(rs[1]), lkT1))
		}
		res = append(res, translate(string(rs[0]), lkT2))
		rs = rs[2:]
	}

	iv := -2 // sentinel (Python None): never equals i-1 for i>=0
	n := len(rs)
	for i, c := range rs {
		switch {
		case c == 'ะ' || c == 'ั' || c == 'ิ' || c == 'ี': // 0e30,0e31,0e34,0e35 — separator only
			iv = i
			res = append(res, "")
		case c == 'า' || c == 'ึ' || c == 'ื' || c == 'ู' || c == 'ๅ': // 0e32,0e36,0e37,0e39,0e45
			iv = i
			res = append(res, translate(string(c), lkT2))
		case c == 'ุ': // 0e38
			iv = i
			if i == 0 || (rs[i-1] != 'ต' && rs[i-1] != 'ธ') {
				res = append(res, translate(string(c), lkT2))
			} else {
				res = append(res, "")
			}
		case c == 'ห' || c == 'อ': // 0e2b, 0e2d
			if i+1 < n && (rs[i+1] == 'ึ' || rs[i+1] == 'ื' || rs[i+1] == 'ุ' || rs[i+1] == 'ู') {
				res = append(res, translate(string(c), lkT2))
			}
		case c == 'ย' || c == 'ร' || c == 'ฤ' || c == 'ฦ' || c == 'ว': // 0e22,23,24,26,27
			if iv == i-1 || (i+1 < n && (rs[i+1] == 'ึ' || rs[i+1] == 'ื' || rs[i+1] == 'ุ' || rs[i+1] == 'ู')) {
				res = append(res, translate(string(c), lkT2))
			}
		default:
			res = append(res, translate(string(c), lkT2))
		}
	}

	if len(res) == 0 {
		return firstRunes("0000", 5)
	}
	res2 := []string{res[0]}
	for i := 1; i < len(res); i++ {
		if res[i] != res[i-1] {
			res2 = append(res2, res[i])
		}
	}
	return firstRunes(strings.Join(res2, "")+"0000", 5)
}
