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
	"sync"
	"unicode"
	"unicode/utf8"

	"github.com/sukitss/thai-nlp-go/dict"
)

const maxGraphSize = 50

// scratch holds onecut's reusable buffers, pooled so repeated calls (across a
// corpus) allocate almost nothing at steady state. A Segmenter never stores
// scratch, so plain Segmenters remain safe for concurrent use — each call
// borrows its own scratch from the pool.
type scratch struct {
	parent  []int
	visited []int // generation-marked; gen persists so no per-call reset is needed
	queue   []int
	pathBuf []int
	lenBuf  []int
	psbuf   []int
	// candidate graph as a generation-marked linked list (preserves edge
	// insertion order, which BFS relies on): adjHead/adjTail index into
	// edgeTo/edgeNext; a node has edges iff adjGen[node] == the current gen.
	adjHead  []int
	adjTail  []int
	adjGen   []int
	edgeTo   []int
	edgeNext []int
	gen      int
	adjG     int
}

var scratchPool = sync.Pool{New: func() any { return &scratch{} }}

func (sc *scratch) prepare(n int) {
	if cap(sc.parent) < n+1 {
		sc.parent = make([]int, n+1)
		sc.visited = make([]int, n+1) // zeroed; gen (>0, persisted) marks visits
		sc.adjHead = make([]int, n+1)
		sc.adjTail = make([]int, n+1)
		sc.adjGen = make([]int, n+1) // zeroed; adjG (>0, persisted) marks edges
	} else {
		sc.parent = sc.parent[:n+1]
		sc.visited = sc.visited[:n+1]
		sc.adjHead = sc.adjHead[:n+1]
		sc.adjTail = sc.adjTail[:n+1]
		sc.adjGen = sc.adjGen[:n+1]
	}
	sc.queue = sc.queue[:0]
	sc.pathBuf = sc.pathBuf[:0]
	sc.lenBuf = sc.lenBuf[:0]
	sc.edgeTo = sc.edgeTo[:0]
	sc.edgeNext = sc.edgeNext[:0]
	sc.adjG++ // fresh generation so no stale node appears to have edges
}

// TCCEngine reports Thai Character Cluster boundaries. TCC satisfies it.
type TCCEngine interface {
	PosArray(text []rune) []bool
}

// Segmenter tokenizes Thai text. Create one with New or NewDefault.
//
// Concurrency: a plain Segmenter (from New/NewDefault) is safe for concurrent
// use when its dictionary is read-only and stateless — the shared default
// (dict.Default) and dict.FlatTrie/Trie are. A Segmenter from Session or
// SessionWithDict is NOT safe for concurrent use, because its overlay keeps
// per-lookup scratch buffers; create one per goroutine instead. The underlying
// base dictionary and overlay *dict.Trie are read-only and may be shared freely.
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
// session). It builds a fresh overlay trie every call; if you reuse the same
// word list across many calls, build the trie once and use SessionWithDict.
func (s *Segmenter) Session(words []string) *Segmenter {
	ov := dict.NewTrie()
	for _, w := range words {
		ov.Add(w)
	}
	return s.SessionWithDict(ov)
}

// SessionWithDict returns a new Segmenter that recognizes the words in overlay
// on top of the shared base dictionary, reusing a caller-provided (and typically
// caller-cached) *dict.Trie — the overlay is not rebuilt. This is the building
// block for per-user "dynamic" dictionaries: the application owns the cache
// (build a trie per user, keep it in an LRU, evict/persist as it sees fit) while
// the lib stays stateless.
//
//	base, _ := dict.Default()          // shared, loaded once
//	ov := dict.NewTrie()               // build + cache this per user
//	for _, w := range userWords { ov.Add(w) }
//	seg := base... tokenize.New(...).SessionWithDict(ov) // cheap; make one per goroutine
//
// The base dictionary and the overlay *dict.Trie are read-only and safe to share
// across goroutines; the returned Segmenter is not (see the type doc).
func (s *Segmenter) SessionWithDict(overlay *dict.Trie) *Segmenter {
	return &Segmenter{
		d:       NewOverlayDict(s.d, overlay),
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
	validPoss := s.tcc.PosArray(text)

	// Borrow reusable buffers from the pool (returned before we exit). Buffers
	// are onecut-local, so a plain Segmenter stays safe for concurrent use.
	sc := scratchPool.Get().(*scratch)
	sc.prepare(n)
	parent := sc.parent
	visited := sc.visited
	queue := sc.queue
	pathBuf := sc.pathBuf
	lenBuf := sc.lenBuf
	adjHead := sc.adjHead
	adjTail := sc.adjTail
	adjGen := sc.adjGen
	edgeTo := sc.edgeTo
	edgeNext := sc.edgeNext
	adjG := sc.adjG

	graphSize := 0
	endPos := 0
	ps := &posSet{s: append(sc.psbuf[:0], 0)}

	for ps.peekMin() < n {
		beginPos := ps.popMin()
		lenBuf = s.d.PrefixLens(text, beginPos, lenBuf)
		for _, L := range lenBuf {
			cand := beginPos + L
			if validPoss[cand] {
				// append edge beginPos -> cand (keep insertion order)
				idx := len(edgeTo)
				edgeTo = append(edgeTo, cand)
				edgeNext = append(edgeNext, -1)
				if adjGen[beginPos] != adjG {
					adjGen[beginPos] = adjG
					adjHead[beginPos] = idx
				} else {
					edgeNext[adjTail[beginPos]] = idx
				}
				adjTail[beginPos] = idx
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
			goal := ps.peekMin()
			// BFS for the first (shortest) path endPos -> goal, using reusable
			// buffers + generation-marked visited (no per-call allocation).
			sc.gen++
			g := sc.gen
			queue = queue[:0]
			queue = append(queue, endPos)
			visited[endPos] = g
			for qi := 0; qi < len(queue); qi++ {
				v := queue[qi]
				stop := false
				if adjGen[v] == adjG {
					for e := adjHead[v]; e != -1; e = edgeNext[e] {
						pos := edgeTo[e]
						if visited[pos] == g {
							continue
						}
						visited[pos] = g
						parent[pos] = v
						if pos == goal {
							stop = true
							break
						}
						queue = append(queue, pos)
					}
				}
				if stop {
					break
				}
			}
			// reset the candidate graph for the next window
			graphSize = 0
			adjG++
			edgeTo = edgeTo[:0]
			edgeNext = edgeNext[:0]
			// reconstruct goal..endPos backward, then emit spans forward
			pathBuf = pathBuf[:0]
			for cur := goal; cur != endPos; cur = parent[cur] {
				pathBuf = append(pathBuf, cur)
			}
			prev := endPos
			for i := len(pathBuf) - 1; i >= 0; i-- {
				spans = append(spans, span{prev, pathBuf[i]})
				prev = pathBuf[i]
			}
			endPos = prev
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
			adjG++
			edgeTo = edgeTo[:0]
			edgeNext = edgeNext[:0]
			spans = append(spans, span{beginPos, endPos})
			ps.push(endPos)
		}
	}
	// Return grown buffers to the pool for reuse by later calls.
	sc.queue = queue
	sc.pathBuf = pathBuf
	sc.lenBuf = lenBuf
	sc.psbuf = ps.s
	sc.edgeTo = edgeTo
	sc.edgeNext = edgeNext
	sc.adjG = adjG
	scratchPool.Put(sc)
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

// MaxSubwords caps the fine subwords emitted per token (recall vs index size).
const MaxSubwords = 16

// Subwords returns the dictionary words strictly inside word (the fine field of
// a coarse/fine keyword index). See dict.Subwords. Empty when word has no
// smaller in-dictionary pieces.
func (s *Segmenter) Subwords(word string) []string {
	return dict.Subwords(s.d, word, MaxSubwords)
}
