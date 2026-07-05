package multi

import (
	"strings"
	"unicode/utf8"

	"github.com/sukitss/thai-nlp-go/cjk"
	"github.com/sukitss/thai-nlp-go/dict"
	"github.com/sukitss/thai-nlp-go/en"
	"github.com/sukitss/thai-nlp-go/jp"
	"github.com/sukitss/thai-nlp-go/kr"
	"github.com/sukitss/thai-nlp-go/normalize"
	"github.com/sukitss/thai-nlp-go/stopwords"
	"github.com/sukitss/thai-nlp-go/token"
	"github.com/sukitss/thai-nlp-go/tokenize"
)

// HanMode selects the tokenizer for an ambiguous all-kanji CJK run (no kana —
// could be Chinese or Japanese). The zero value ("") falls back to the mutable
// package global DefaultHan for backward compatibility with Segment; new code
// should set it explicitly so the choice is carried by the Analyzer value, not
// by hidden global state.
type HanMode string

const (
	HanChinese  HanMode = "cn" // ambiguous all-kanji runs → cjk (Chinese)
	HanJapanese HanMode = "jp" // ambiguous all-kanji runs → jp (Japanese)
)

// Analyzer is a configured, per-tenant analysis pipeline over mixed-language
// text: normalize.Normalize → optional Thai-digit folding → script routing
// (identical to Segment's) → per-run tokenizers → term post-filters. A RAG
// engine must analyze documents at ingest and queries at search with EXACTLY
// the same pipeline (any asymmetry silently breaks sparse retrieval); holding
// the whole configuration in one value makes that a matter of sharing the
// value.
//
// The zero value behaves like Segment (no overlay, no stop words, no folding,
// DefaultHan routing). An Analyzer is plain immutable-by-convention data:
// configure it once, then it is SAFE FOR CONCURRENT USE — share one per tenant
// across request goroutines. (Per-call tokenizer sessions are created
// internally; nothing is mutated on the Analyzer itself.)
//
// Deterministic: same input + same configuration → same output. The only
// global state consulted is the documented DefaultHan fallback when Han is
// left at its zero value.
type Analyzer struct {
	// Overlay is a per-tenant dictionary overlay (e.g. a translator glossary)
	// recognized on Thai runs on top of the shared base dictionary; nil means
	// base only. The *dict.Trie is read-only here and may be shared freely.
	Overlay *dict.Trie

	// Stop drops matching terms from Terms/AppendTerms output; nil keeps all.
	// It does NOT affect Tokens (the highlighting path). Matching is exact —
	// when LowerLatin is set the check runs on the LOWERCASED term, so a set
	// containing "the" also catches "The" (see LowerLatin's acronym caveat).
	Stop *stopwords.Set

	// LowerLatin lowercases Latin-route tokens (Latin-script runs and the
	// digit/symbol fallback route — everything en.Cut handles, like
	// en.CutLower) in Terms/AppendTerms. Tokens keeps original text. Caveat:
	// folding erases the case distinction the stopwords package was designed
	// around ("it" is a stop word, the acronym "IT" is not) — with LowerLatin
	// on, both fold to "it" and both are dropped when Stop contains it. The
	// tenant chooses the trade-off.
	LowerLatin bool

	// Han routes ambiguous all-kanji runs (HanChinese/HanJapanese). When set,
	// DefaultHan is never consulted; the zero value preserves the old
	// DefaultHan behavior. Any other non-empty value routes like HanChinese.
	Han HanMode

	// UseDP segments Chinese/Japanese runs with CutDP (DAG + dynamic
	// programming over word weights) instead of greedy longest-match Cut.
	UseDP bool

	// FoldThaiDigits applies normalize.DigitsToArabic after Normalize, so
	// "๕" indexes as "5". Off by default (Normalize alone never folds digits).
	FoldThaiDigits bool

	// FoldWidth applies normalize.FoldForIndex (full/half-width folding + NFC)
	// so CJK/Latin variants unify: full-width "ＡＰＩ" matches "API", half-width
	// "ｶﾀｶﾅ" matches "カタカナ", NFD Korean matches NFC. Off by default; a
	// multilingual index should turn it on (ingest AND query, symmetrically).
	// It does not fold simplified↔traditional Chinese or hiragana↔katakana.
	FoldWidth bool
}

// Terms runs the full pipeline and returns the filtered terms for indexing or
// querying: normalized, routed, tokenized per run, then stop words dropped and
// Latin case folded per the configuration. Returns nil when nothing survives.
func (a *Analyzer) Terms(text string) []string {
	norm := a.normalized(text)
	if norm == "" {
		return nil
	}
	var out []string
	th := a.thaiSegmenter()
	forEachRunOff(norm, func(l lang, start, end int, hasKana bool) {
		for _, t := range a.cutRun(th, l, norm[start:end], hasKana) {
			if t = a.filterTerm(l, t); t != "" {
				out = append(out, t)
			}
		}
	})
	return out
}

// AppendTerms appends Terms(text) joined by sep to dst and returns the
// extended slice — the indexing path. Reuse dst (dst[:0]) to amortize the
// output buffer; what still allocates is each run's tokenizer returning its
// tokens as []string internally (same trade-off as AppendBytes) plus, when
// Overlay is set, the per-call Thai session.
func (a *Analyzer) AppendTerms(dst []byte, text string, sep byte) []byte {
	norm := a.normalized(text)
	if norm == "" {
		return dst
	}
	first := true
	th := a.thaiSegmenter()
	forEachRunOff(norm, func(l lang, start, end int, hasKana bool) {
		for _, t := range a.cutRun(th, l, norm[start:end], hasKana) {
			if t = a.filterTerm(l, t); t == "" {
				continue
			}
			if !first {
				dst = append(dst, sep)
			}
			first = false
			dst = append(dst, t...)
		}
	})
	return dst
}

// Tokens runs the SAME pipeline as Terms but returns every token with byte
// offsets, plus the normalized text those offsets index into:
// norm[t.Start:t.End] == t.Text always holds. This is the highlighting /
// entity-offset path, so NO stop-word filtering and NO lowercasing is applied
// — filtering here would hide matches.
//
// Offsets are NOT offsets into the caller's original text argument: the
// pipeline rewrites the string (Normalize collapses spaces, reorders marks;
// FoldThaiDigits shrinks ๕ to 5), so positions only make sense in the
// returned norm. Highlight against norm, or store norm alongside the offsets.
func (a *Analyzer) Tokens(text string) (norm string, toks []token.Token) {
	norm = a.normalized(text)
	if norm == "" {
		return norm, nil
	}
	th := a.thaiSegmenter()
	forEachRunOff(norm, func(l lang, start, end int, hasKana bool) {
		run := norm[start:end]
		switch l {
		case thai:
			n := len(toks)
			toks = th.AppendTokensNoWS(toks, run)
			for k := n; k < len(toks); k++ {
				toks[k].Start += start
				toks[k].End += start
			}
		case cjkHan:
			var rt []token.Token
			switch {
			case hasKana || a.hanJapanese():
				if a.UseDP {
					rt = jp.TokensDP(run)
				} else {
					rt = jp.Tokens(run)
				}
			case a.UseDP:
				rt = cjk.TokensDP(run)
			default:
				rt = cjk.Tokens(run)
			}
			toks = appendShifted(toks, rt, start)
		case hangul:
			toks = appendShifted(toks, kr.Tokens(run), start)
		default: // latin + digits/symbols/other letters
			toks = appendShifted(toks, en.Tokens(run), start)
		}
	})
	return norm, toks
}

// normalized applies the shared front of the pipeline: Normalize, then the
// optional Thai-digit fold.
func (a *Analyzer) normalized(text string) string {
	text = normalize.Normalize(text)
	if a.FoldThaiDigits {
		text = normalize.DigitsToArabic(text)
	}
	if a.FoldWidth {
		text = normalize.FoldForIndex(text)
	}
	return text
}

// thaiSegmenter returns the Segmenter for this call's Thai runs: the shared
// default when there is no overlay, else a fresh session per call. A session
// is two tiny structs (Segmenter + overlay wrapper) that never leave the call
// frame — measured ~4 ns / 0 heap allocations under escape analysis, and the
// full per-tenant configuration lands within ~2% of the zero config on the
// mixed benchmark — so pooling would buy nothing. Per-call construction keeps
// the Analyzer free of mutable state; a session's lookup scratch is
// single-threaded by design, so sharing one across request goroutines would
// need a lock right in the tokenizer hot path.
func (a *Analyzer) thaiSegmenter() *tokenize.Segmenter {
	if a.Overlay == nil {
		return thai_()
	}
	return thai_().SessionWithDict(a.Overlay)
}

// hanJapanese reports whether an ambiguous all-kanji run routes to Japanese.
func (a *Analyzer) hanJapanese() bool {
	if a.Han != "" {
		return a.Han == HanJapanese // explicit: DefaultHan is not consulted
	}
	return DefaultHan == "jp"
}

// cutRun tokenizes one script run with this Analyzer's configuration — the
// same routing as tokenizeRun, minus the DefaultHan global when Han is set.
func (a *Analyzer) cutRun(th *tokenize.Segmenter, l lang, run string, hasKana bool) []string {
	switch l {
	case thai:
		return th.SegmentNoWS(run)
	case cjkHan:
		if hasKana || a.hanJapanese() {
			if a.UseDP {
				return jp.CutDP(run)
			}
			return jp.Cut(run)
		}
		if a.UseDP {
			return cjk.CutDP(run)
		}
		return cjk.Cut(run)
	case hangul:
		return kr.Cut(run)
	default:
		return en.Cut(run) // digits/symbols/other letters: whitespace-ish split
	}
}

// filterTerm applies the Terms/AppendTerms post-filters to one token and
// returns the surviving term, or "" to drop it. Order matters and is part of
// the contract: LowerLatin folds FIRST, then Stop is checked against the
// folded term.
func (a *Analyzer) filterTerm(l lang, t string) string {
	if a.LowerLatin && (l == latin || l == other) {
		t = strings.ToLower(t)
	}
	if a.Stop != nil && a.Stop.IsStopword(t) {
		return ""
	}
	return t
}

// appendShifted appends run-relative tokens to dst with their offsets shifted
// into whole-string coordinates.
func appendShifted(dst, toks []token.Token, off int) []token.Token {
	for _, t := range toks {
		t.Start += off
		t.End += off
		dst = append(dst, t)
	}
	return dst
}

// forEachRunOff is forEachRun with byte offsets: it groups text into the same
// maximal same-language runs (Common characters attach to the current run) and
// calls fn with each run's half-open byte range, so run text can be taken as a
// raw slice text[start:end] and token offsets composed into whole-string
// positions. Decoding runes in place yields the exact rune sequence that
// forEachRun sees via []rune, so both walkers split identically; the only
// difference is that a raw slice preserves invalid UTF-8 bytes where
// forEachRun's string(rs[i:j]) substitutes U+FFFD — which is what the offset
// guarantee norm[Start:End] == Text requires.
func forEachRunOff(text string, fn func(l lang, start, end int, hasKana bool)) {
	i := 0
	for i < len(text) {
		r, sz := utf8.DecodeRuneInString(text[i:])
		if isCommon(r) {
			i += sz
			continue
		}
		l := classify(r)
		j := i
		hasKana := false
		for j < len(text) {
			r2, sz2 := utf8.DecodeRuneInString(text[j:])
			if isCommon(r2) {
				j += sz2
				continue
			}
			if isKana(r2) && l == cjkHan {
				hasKana = true
				j += sz2
				continue
			}
			if classify(r2) != l {
				break
			}
			j += sz2
		}
		fn(l, i, j, hasKana)
		i = j
	}
}
