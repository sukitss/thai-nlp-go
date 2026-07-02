package tokenize

// Thai Character Cluster (TCC) segmentation — hand-coded, zero-allocation port
// of PyThaiNLP's pythainlp.tokenize.tcc_p._RE_TCC (29 regex patterns).
//
// The original uses regexes with look-ahead (which Go's RE2 can't express). We
// encode each pattern as a small combinator instead: alternation is
// leftmost-first (matching Python's `re` alternation) and every optional inside
// a TCC pattern is followed by a disjoint class, so greedy-no-backtrack is
// correct. The result matches PyThaiNLP exactly and allocates nothing per match.
// A regexp2-based oracle in the test suite verifies this equivalence.

type matcher func(t []rune, i int) int // returns next index, or -1 on no match

// ----- character classes (codepoints follow _RE_TCC) -----
func isCons(r rune) bool      { return r >= 0x0E01 && r <= 0x0E2E } // ก-ฮ
func isTone(r rune) bool      { return r >= 0x0E48 && r <= 0x0E4B } // ่-๋
func inEToAi(r rune) bool     { return r >= 0x0E40 && r <= 0x0E44 } // เ-ไ
func inSaraRange(r rune) bool { return r >= 0x0E30 && r <= 0x0E39 } // ะ-ู

func anyOf(rs ...rune) func(rune) bool {
	return func(r rune) bool {
		for _, x := range rs {
			if r == x {
				return true
			}
		}
		return false
	}
}

// ----- combinators (built once at init; no allocation during matching) -----
func lit(r rune) matcher {
	return func(t []rune, i int) int {
		if i < len(t) && t[i] == r {
			return i + 1
		}
		return -1
	}
}

func cls(test func(rune) bool) matcher {
	return func(t []rune, i int) int {
		if i < len(t) && test(t[i]) {
			return i + 1
		}
		return -1
	}
}

func seq(ms ...matcher) matcher {
	return func(t []rune, i int) int {
		for _, m := range ms {
			i = m(t, i)
			if i < 0 {
				return -1
			}
		}
		return i
	}
}

func opt(m matcher) matcher {
	return func(t []rune, i int) int {
		if j := m(t, i); j >= 0 {
			return j
		}
		return i
	}
}

// look — look-ahead (?=[เ-ไก-ฮ]|$): next char ∈ {เ-ไ, ก-ฮ} or end (zero-width).
func look() matcher {
	return func(t []rune, i int) int {
		if i >= len(t) || inEToAi(t[i]) || isCons(t[i]) {
			return i
		}
		return -1
	}
}

// rune literals used by the patterns
const (
	rE    = 0x0E40 // เ
	rAE   = 0x0E41 // แ
	rO    = 0x0E42 // โ
	rI    = 0x0E34 // ิ
	rII   = 0x0E35 // ี
	rUE   = 0x0E36 // ึ
	rUEUE = 0x0E37 // ื
	rU    = 0x0E38 // ุ
	rUU   = 0x0E39 // ู
	rA    = 0x0E30 // ะ
	rAN   = 0x0E31 // ั
	rAA   = 0x0E32 // า
	rAM   = 0x0E33 // ำ
	rTH   = 0x0E47 // ็
	rKA   = 0x0E4C // ์
	rYO   = 0x0E22 // ย
	rWO   = 0x0E27 // ว
	rRO   = 0x0E23 // ร
	rOO   = 0x0E2D // อ
	rHO   = 0x0E2B // ห
	rKO   = 0x0E01 // ก
)

// TCC finds Thai Character Cluster boundaries. It is stateless and safe to
// share across goroutines.
type TCC struct {
	pats []matcher
}

// NewTCC builds the TCC matcher.
func NewTCC() *TCC {
	C := cls(isCons)
	T := cls(isTone)
	clsUUI := cls(anyOf(rUU, rU, rI))       // [ูุิ]
	clsIIUU := cls(anyOf(rI, rII, rU, rUU)) // [ิีุู]
	clsAnUe := cls(anyOf(rAN, rUEUE))       // [ัื]
	clsUIA := cls(anyOf(rU, rI, rA))        // [ุิะ]
	clsSara := cls(inSaraRange)             // [ะ-ู]
	clsAAmA := cls(anyOf(rA, rAA, rAM))     // [ะาำ]
	// tail = ([ก-ฮ][ก-ฮ]?[ูุิ]?[์])?
	tail := opt(seq(C, opt(C), opt(clsUUI), lit(rKA)))

	p := []matcher{
		seq(lit(rE), C, lit(rTH), C, tail),                                // 0  เc็ck
		seq(lit(rE), C, C, opt(T), lit(rAA), lit(rA), tail),               // 1  เccตาะk
		seq(lit(rE), C, C, lit(rII), opt(T), lit(rYO), lit(rA), tail),     // 2  เccีtยะk
		seq(lit(rE), C, C, lit(rII), opt(T), lit(rYO), look(), tail),      // 3  เccีtย(?=)k
		seq(lit(rE), C, C, lit(rTH), C, tail),                             // 4  เcc็ck
		seq(lit(rE), C, lit(rI), C, lit(rKA), C, tail),                    // 5  เcิc์ck
		seq(lit(rE), C, lit(rI), opt(T), C, tail),                         // 6  เcิtck
		seq(lit(rE), C, lit(rII), opt(T), lit(rYO), opt(lit(rA)), tail),   // 7  เcีtยะ?k
		seq(lit(rE), C, lit(rUEUE), opt(T), lit(rOO), opt(lit(rA)), tail), // 8  เcืtอะ?k
		seq(lit(rE), C, clsIIUU, opt(T), lit(rYO), look(), tail),          // 9  เc[ิีุู]tย(?=)k
		seq(lit(rE), C, opt(T), opt(lit(rAA)), opt(lit(rA)), tail),        // 10 เctา?ะ?k
		seq(C, lit(rAN), opt(T), lit(rWO), lit(rA), tail),                 // 11 cัtวะk
		seq(C, clsAnUe, opt(T), C, opt(clsUIA), tail),                     // 12 c[ัื]tc[ุิะ]?k
		seq(C, cls(anyOf(rI, rU, rUU)), lit(rKA)),                         // 13 c[ิุู]์
		seq(C, clsSara, opt(T), tail),                                     // 14 c[ะ-ู]tk
		seq(C, lit(rRO), lit(rRO), C, lit(rKA)),                           // 15 cรรc์
		seq(C, lit(rTH)),                                                  // 16 c็
		seq(C, opt(T), opt(clsAAmA), tail),                                // 17 ct[ะาำ]?k
		seq(C, tail),                                                      // 18 ck
		seq(lit(rAE), C, lit(rTH), C),                                     // 19 แc็c
		seq(lit(rAE), C, C, lit(rKA)),                                     // 20 แcc์
		seq(lit(rAE), C, opt(T), lit(rA)),                                 // 21 แctะ
		seq(lit(rAE), C, C, lit(rTH), C),                                  // 22 แcc็c
		seq(lit(rAE), C, C, C, lit(rKA)),                                  // 23 แccc์
		seq(lit(rO), C, opt(T), lit(rA)),                                  // 24 โctะ
		seq(cls(inEToAi), C, opt(T)),                                      // 25 [เ-ไ]ct
		seq(lit(rKO), lit(rTH)),                                           // 26 ก็
		seq(lit(rOO), lit(rUE)),                                           // 27 อึ
		seq(lit(rHO), lit(rUE)),                                           // 28 หึ
	}
	return &TCC{pats: p}
}

// PosArray returns a slice of length len(text)+1 that is true at every TCC
// boundary (mirrors PyThaiNLP's tcc_pos_array).
func (f *TCC) PosArray(text []rune) []bool {
	n := len(text)
	arr := make([]bool, n+1)
	p := 0
	for p < n {
		step := 1
		for _, m := range f.pats {
			if j := m(text, p); j > p {
				step = j - p
				break
			}
		}
		p += step
		arr[p] = true
	}
	return arr
}
