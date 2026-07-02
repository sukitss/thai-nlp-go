package tokenize

// newmm word tokenizer — ported from PyThaiNLP's pythainlp.tokenize.newmm
// (_onecut + segment). Maximal matching over a dictionary trie, with candidate
// boundaries constrained to Thai Character Clusters (see tcc.go) and non-Thai
// runs (latin / digits / whitespace / symbols) handled by a dedicated matcher.
//
// The core walk (onecut) is a faithful port; it records token spans rather than
// materializing strings, so callers can render tokens as []string or write them
// straight into a []byte index with no intermediate allocation.

import (
	"unicode"
	"unicode/utf8"

	"github.com/sukitss/thai-nlp-go/dict"
)

const maxGraphSize = 50

// TCCEngine reports Thai Character Cluster boundaries. TCC satisfies it.
type TCCEngine interface {
	PosArray(text []rune) []bool
}

// Segmenter tokenizes Thai text. Create one with New or NewDefault. A Segmenter
// is safe for concurrent use as long as its dictionary is (the shared default
// is). Session derives a cheap per-session copy with extra dictionary words.
type Segmenter struct {
	d       dict.Prefixer
	tcc     TCCEngine
	nonThai func(text []rune, pos int) int
}

// New returns a Segmenter using d as its dictionary and the built-in (fast,
// allocation-free) TCC engine. Use NewDefault for the shared embedded dictionary.
func New(d dict.Prefixer) *Segmenter {
	return &Segmenter{d: d, tcc: NewTCC(), nonThai: nonThaiEndFast}
}

// NewDefault returns a Segmenter backed by the shared default dictionary
// (dict.Default) — loaded once and shared across the whole process — plus the
// built-in TCC engine. This is the ready-to-use constructor.
func NewDefault() (*Segmenter, error) {
	d, err := dict.Default()
	if err != nil {
		return nil, err
	}
	return New(d), nil
}

// Session returns a new Segmenter that shares this one's base dictionary and
// engines but also recognizes the given extra words (e.g. custom names for one
// session). The base is untouched, so many sessions can run concurrently over
// the same Segmenter.
func (s *Segmenter) Session(words []string) *Segmenter {
	ov := dict.NewTrie()
	for _, w := range words {
		ov.Add(w)
	}
	return &Segmenter{
		d:       NewOverlayDict(s.d, ov),
		tcc:     s.tcc,
		nonThai: s.nonThai,
	}
}

// span is a half-open [s,e) rune range of the input covered by one token.
type span struct{ s, e int }

// Segment returns raw tokens (equivalent to newmm.segment /
// word_tokenize(..., keep_whitespace=True)): whitespace tokens are preserved.
func (s *Segmenter) Segment(text string) []string {
	if text == "" {
		return nil
	}
	rs := []rune(text)
	spans := s.onecut(rs)
	out := make([]string, len(spans))
	for i, sp := range spans {
		out[i] = string(rs[sp.s:sp.e])
	}
	return out
}

// SegmentNoWS returns tokens with pure-space tokens dropped and surrounding
// ASCII spaces trimmed — equivalent to word_tokenize(..., keep_whitespace=False).
// This is the "text_seg" form used to feed a BM25/keyword index.
func (s *Segmenter) SegmentNoWS(text string) []string {
	if text == "" {
		return nil
	}
	rs := []rune(text)
	spans := s.onecut(rs)
	out := make([]string, 0, len(spans))
	for _, sp := range spans {
		a, b := trimSpaces(rs, sp.s, sp.e)
		if a < b {
			out = append(out, string(rs[a:b]))
		}
	}
	return out
}

// SegmentBytes tokenizes like SegmentNoWS but writes the tokens straight into a
// freshly allocated []byte, joined by sep — no per-token string allocation.
// Ideal for building an inverted index / keyword field.
func (s *Segmenter) SegmentBytes(text string, sep byte) []byte {
	return s.AppendBytes(nil, text, sep)
}

// AppendBytes appends SegmentNoWS tokens (joined by sep) to dst and returns the
// extended slice, reusing dst's capacity across calls for zero steady-state
// allocation.
func (s *Segmenter) AppendBytes(dst []byte, text string, sep byte) []byte {
	if text == "" {
		return dst
	}
	rs := []rune(text)
	spans := s.onecut(rs)
	first := true
	for _, sp := range spans {
		a, b := trimSpaces(rs, sp.s, sp.e)
		if a >= b {
			continue
		}
		if !first {
			dst = append(dst, sep)
		}
		first = false
		for i := a; i < b; i++ {
			dst = utf8.AppendRune(dst, rs[i])
		}
	}
	return dst
}

// ---------- core walk (faithful port of PyThaiNLP _onecut + segment) ----------

// onecut returns the token spans covering text. Spans are contiguous and cover
// the whole input: spans[0].s == 0, spans[i].s == spans[i-1].e, last.e == len.
func (s *Segmenter) onecut(text []rune) []span {
	n := len(text)
	spans := make([]span, 0, n/3+1)
	graph := make(map[int][]int)
	graphSize := 0
	validPoss := s.tcc.PosArray(text)

	ps := &posSet{s: []int{0}}
	endPos := 0
	lenBuf := make([]int, 0, 16)

	for ps.peekMin() < n {
		beginPos := ps.popMin()
		lens := s.d.PrefixLens(text, beginPos, lenBuf)
		for _, L := range lens {
			cand := beginPos + L
			if validPoss[cand] {
				graph[beginPos] = append(graph[beginPos], cand)
				graphSize++
				if !ps.contains(cand) {
					ps.push(cand)
				}
				if graphSize > maxGraphSize {
					break
				}
			}
		}

		switch ps.len() {
		case 1:
			path := bfsFirstPath(graph, endPos, ps.peekMin())
			graphSize = 0
			clear(graph)
			for _, pos := range path[1:] {
				spans = append(spans, span{endPos, pos})
				endPos = pos
			}
		case 0:
			if e := s.nonThai(text, beginPos); e >= 0 {
				endPos = e
			} else {
				found := false
				for pos := beginPos + 1; pos < n; pos++ {
					if validPoss[pos] {
						hasWord := false
						for _, L := range s.d.PrefixLens(text, pos, lenBuf) {
							if validPoss[pos+L] && !isThaiTwoChars(text, pos, pos+L) {
								hasWord = true
								break
							}
						}
						if hasWord {
							endPos = pos
							found = true
							break
						}
						if s.nonThai(text, pos) >= 0 {
							endPos = pos
							found = true
							break
						}
					}
				}
				if !found {
					endPos = n
				}
			}
			graphSize = 0
			clear(graph)
			spans = append(spans, span{beginPos, endPos})
			ps.push(endPos)
		}
	}
	return spans
}

// ---------- non-Thai matcher (hand-coded _PAT_NONTHAI.match) ----------

func isThaiBlock(r rune) bool { return r >= 0x0E00 && r <= 0x0E7F }
func isLatin(r rune) bool {
	return r == '-' || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z')
}

// isDigit matches Python's re `\d` (Unicode category Nd, incl. Thai ๐-๙).
func isDigit(r rune) bool { return unicode.IsDigit(r) }

// nonThaiEndFast is the hand-coded _PAT_NONTHAI.match(text,pos): leftmost-first,
// returning the end index or -1.
func nonThaiEndFast(t []rune, pos int) int {
	n := len(t)
	if pos >= n {
		return -1
	}
	c := t[pos]
	switch {
	case isLatin(c): // [-a-zA-Z]+
		i := pos + 1
		for i < n && isLatin(t[i]) {
			i++
		}
		return i
	case isDigit(c): // \d+([,.]\d+)*
		i := pos + 1
		for i < n && isDigit(t[i]) {
			i++
		}
		for i+1 < n && (t[i] == ',' || t[i] == '.') && isDigit(t[i+1]) {
			i++
			for i < n && isDigit(t[i]) {
				i++
			}
		}
		return i
	case c == ' ' || c == '\t': // [ \t]+
		i := pos + 1
		for i < n && (t[i] == ' ' || t[i] == '\t') {
			i++
		}
		return i
	case c == '\n': // \r?\n (the \n case)
		return pos + 1
	case c == '\r': // \r?\n (must be followed by \n)
		if pos+1 < n && t[pos+1] == '\n' {
			return pos + 2
		}
		return -1 // lone \r: matches no rule
	default: // [^฀-๿ \t\r\n]+
		if isThaiBlock(c) {
			return -1
		}
		i := pos + 1
		for i < n {
			r := t[i]
			if isThaiBlock(r) || r == ' ' || r == '\t' || r == '\r' || r == '\n' {
				break
			}
			i++
		}
		return i
	}
}

// isThaiTwoChars mirrors _PAT_THAI_TWOCHARS = "[ก-ฮ]{,2}$": ≤2 Thai consonants.
func isThaiTwoChars(text []rune, start, end int) bool {
	if end-start > 2 {
		return false
	}
	for i := start; i < end; i++ {
		if text[i] < 0x0E01 || text[i] > 0x0E2E { // ก..ฮ
			return false
		}
	}
	return true
}

// bfsFirstPath mirrors next(_bfs_paths_graph(...)): the first path start→goal.
func bfsFirstPath(graph map[int][]int, start, goal int) []int {
	visited := map[int]bool{start: true}
	type item struct {
		v    int
		path []int
	}
	queue := []item{{start, []int{start}}}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		for _, pos := range graph[cur.v] {
			if pos == goal {
				np := make([]int, len(cur.path)+1)
				copy(np, cur.path)
				np[len(cur.path)] = pos
				return np
			} else if !visited[pos] {
				visited[pos] = true
				np := make([]int, len(cur.path)+1)
				copy(np, cur.path)
				np[len(cur.path)] = pos
				queue = append(queue, item{pos, np})
			}
		}
	}
	return nil
}

// ---------- small helpers ----------

// trimSpaces trims leading/trailing ASCII spaces (matching str.strip(" ")).
func trimSpaces(text []rune, s, e int) (int, int) {
	for s < e && text[s] == ' ' {
		s++
	}
	for e > s && text[e-1] == ' ' {
		e--
	}
	return s, e
}

// posSet — a sorted-unique ascending set that emulates PyThaiNLP's min-heap of
// candidate positions (which stays unique by construction).
type posSet struct{ s []int }

func (p *posSet) peekMin() int { return p.s[0] }
func (p *posSet) popMin() int {
	v := p.s[0]
	p.s = p.s[1:]
	return v
}
func (p *posSet) len() int { return len(p.s) }
func (p *posSet) contains(v int) bool {
	for _, x := range p.s {
		if x == v {
			return true
		}
	}
	return false
}
func (p *posSet) push(v int) {
	i := 0
	for i < len(p.s) && p.s[i] < v {
		i++
	}
	if i < len(p.s) && p.s[i] == v {
		return
	}
	p.s = append(p.s, 0)
	copy(p.s[i+1:], p.s[i:])
	p.s[i] = v
}
