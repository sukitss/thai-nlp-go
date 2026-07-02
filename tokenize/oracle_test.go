package tokenize

// Test-only oracle: the original regexp2 (.NET-style, look-ahead) implementation
// of TCC and the non-Thai matcher, kept as a reference to prove the hand-coded
// fast path is equivalent. Because it lives in a _test.go file, the regexp2
// dependency never reaches importers of this library.

import (
	"github.com/dlclark/regexp2"
	"github.com/sukitss/thai-nlp-go/dict"
)

// tccOraclePatterns is the expanded pythainlp.tokenize.tcc_p._RE_TCC (29 rules,
// order matters). #3 and #9 use look-ahead, which RE2 cannot express.
var tccOraclePatterns = []string{
	`เ[ก-ฮ]็[ก-ฮ]([ก-ฮ][ก-ฮ]?[ูุิ]?[์])?`,
	`เ[ก-ฮ][ก-ฮ][่-๋]?าะ([ก-ฮ][ก-ฮ]?[ูุิ]?[์])?`,
	`เ[ก-ฮ][ก-ฮ]ี[่-๋]?ยะ([ก-ฮ][ก-ฮ]?[ูุิ]?[์])?`,
	`เ[ก-ฮ][ก-ฮ]ี[่-๋]?ย(?=[เ-ไก-ฮ]|$)([ก-ฮ][ก-ฮ]?[ูุิ]?[์])?`,
	`เ[ก-ฮ][ก-ฮ]็[ก-ฮ]([ก-ฮ][ก-ฮ]?[ูุิ]?[์])?`,
	`เ[ก-ฮ]ิ[ก-ฮ]์[ก-ฮ]([ก-ฮ][ก-ฮ]?[ูุิ]?[์])?`,
	`เ[ก-ฮ]ิ[่-๋]?[ก-ฮ]([ก-ฮ][ก-ฮ]?[ูุิ]?[์])?`,
	`เ[ก-ฮ]ี[่-๋]?ยะ?([ก-ฮ][ก-ฮ]?[ูุิ]?[์])?`,
	`เ[ก-ฮ]ื[่-๋]?อะ?([ก-ฮ][ก-ฮ]?[ูุิ]?[์])?`,
	`เ[ก-ฮ][ิีุู][่-๋]?ย(?=[เ-ไก-ฮ]|$)([ก-ฮ][ก-ฮ]?[ูุิ]?[์])?`,
	`เ[ก-ฮ][่-๋]?า?ะ?([ก-ฮ][ก-ฮ]?[ูุิ]?[์])?`,
	`[ก-ฮ]ั[่-๋]?วะ([ก-ฮ][ก-ฮ]?[ูุิ]?[์])?`,
	`[ก-ฮ][ัื][่-๋]?[ก-ฮ][ุิะ]?([ก-ฮ][ก-ฮ]?[ูุิ]?[์])?`,
	`[ก-ฮ][ิุู]์`,
	`[ก-ฮ][ะ-ู][่-๋]?([ก-ฮ][ก-ฮ]?[ูุิ]?[์])?`,
	`[ก-ฮ]รร[ก-ฮ]์`,
	`[ก-ฮ]็`,
	`[ก-ฮ][่-๋]?[ะาำ]?([ก-ฮ][ก-ฮ]?[ูุิ]?[์])?`,
	`[ก-ฮ]([ก-ฮ][ก-ฮ]?[ูุิ]?[์])?`,
	`แ[ก-ฮ]็[ก-ฮ]`,
	`แ[ก-ฮ][ก-ฮ]์`,
	`แ[ก-ฮ][่-๋]?ะ`,
	`แ[ก-ฮ][ก-ฮ]็[ก-ฮ]`,
	`แ[ก-ฮ][ก-ฮ][ก-ฮ]์`,
	`โ[ก-ฮ][่-๋]?ะ`,
	`[เ-ไ][ก-ฮ][่-๋]?`,
	`ก็`,
	`อึ`,
	`หึ`,
}

type tccOracle struct{ re *regexp2.Regexp }

func newTCCOracle() *tccOracle {
	pat := `\G(?:`
	for i, p := range tccOraclePatterns {
		if i > 0 {
			pat += "|"
		}
		pat += p
	}
	pat += `)`
	re, err := regexp2.Compile(pat, regexp2.None)
	if err != nil {
		panic(err)
	}
	return &tccOracle{re: re}
}

func (t *tccOracle) PosArray(text []rune) []bool {
	n := len(text)
	arr := make([]bool, n+1)
	p := 0
	for p < n {
		m, _ := t.re.FindRunesMatchStartingAt(text, p)
		step := 1
		if m != nil && m.Index == p && m.Length > 0 {
			step = m.Length
		}
		p += step
		arr[p] = true
	}
	return arr
}

// newOracle builds a Segmenter using the regexp2 TCC + non-Thai matcher.
func newOracle(d dict.Prefixer) *Segmenter {
	nt, err := regexp2.Compile(
		`\G(?:[-a-zA-Z]+|\d+([,.]\d+)*|[ \t]+|\r?\n|[^฀-๿ \t\r\n]+)`,
		regexp2.None)
	if err != nil {
		panic(err)
	}
	nonThai := func(text []rune, pos int) int {
		m, _ := nt.FindRunesMatchStartingAt(text, pos)
		if m != nil && m.Index == pos {
			return pos + m.Length
		}
		return -1
	}
	return &Segmenter{d: d, tcc: newTCCOracle(), nonThai: nonThai}
}
