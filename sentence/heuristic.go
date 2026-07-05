package sentence

import (
	_ "embed"
	"strings"
)

// A middle ground between whitespace splitting (fast but over-splits — Thai
// puts spaces inside sentences too) and the CRF (accurate but ~50x slower and
// needs a model): a linguistically-informed heuristic that keeps whitespace as
// the only split *candidate* but decides each candidate from cheap signals —
// no tokenizer, no model, so it stays close to whitespace speed.
//
// A space (or newline) is a sentence boundary when:
//   - it is a newline (a hard boundary), or
//   - the text before ends with a sentence-ending word (ครับ/ค่ะ/แล้ว/นะ …), or
//   - the text after begins with a sentence-starting word (แต่/ดังนั้น/เมื่อ …), or
//   - the text before ends with ! ? ฯ or an ellipsis.
//
// Otherwise the space is treated as within-sentence and the pieces are joined.
// The ender/starter word lists are PyThaiNLP's crfcut lists (CC-BY-4.0). This is
// a heuristic: it trades the CRF's context modelling for speed, so it is less
// accurate than crfcut but far better than raw whitespace.

//go:embed data/enders.txt
var endersData string

//go:embed data/starters.txt
var startersData string

var enders, starters = wordSet(endersData), wordSet(startersData)

func wordSet(data string) map[string]bool {
	m := map[string]bool{}
	for _, w := range strings.Fields(data) {
		if w != "" && !strings.HasPrefix(w, "#") {
			m[w] = true
		}
	}
	return m
}

// SplitHeuristic splits text into sentences using the ender/starter heuristic
// described in the package. Runs of whitespace are the only split candidates;
// each is kept or dropped by the signals above. Empty segments are dropped.
func SplitHeuristic(text string) []string {
	pieces, hardBreak := splitKeepingBreaks(text)
	if len(pieces) == 0 {
		return nil
	}
	var out []string
	cur := pieces[0]
	for i := 1; i < len(pieces); i++ {
		if hardBreak[i] || endsWithSentPunct(cur) || endsWithEnder(cur) || startsWithStarter(pieces[i]) {
			if s := strings.TrimSpace(cur); s != "" {
				out = append(out, s)
			}
			cur = pieces[i]
		} else {
			cur = cur + " " + pieces[i] // within-sentence space
		}
	}
	if s := strings.TrimSpace(cur); s != "" {
		out = append(out, s)
	}
	return out
}

// splitKeepingBreaks splits on whitespace runs, returning the non-empty pieces
// and, per piece, whether the whitespace before it contained a newline (a hard
// sentence boundary regardless of words).
func splitKeepingBreaks(text string) (pieces []string, hardBreak []bool) {
	i := 0
	n := len(text)
	for i < n {
		// skip whitespace run, noting a newline
		nl := false
		start := i
		for i < n && isSpace(text[i]) {
			if text[i] == '\n' || text[i] == '\r' {
				nl = true
			}
			i++
		}
		leadingWS := i > start
		if i >= n {
			break
		}
		j := i
		for j < n && !isSpace(text[j]) {
			j++
		}
		pieces = append(pieces, text[i:j])
		hardBreak = append(hardBreak, leadingWS && nl)
		i = j
	}
	return pieces, hardBreak
}

func isSpace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r' || b == '\v' || b == '\f'
}

// endsWithEnder reports whether s ends with a sentence-ending word. The check is
// a suffix match against the ender set — cheap and no tokenizer — accepting that
// a rare word coincidentally ending in an ender substring may split early.
func endsWithEnder(s string) bool {
	// strip trailing punctuation first so "แล้ว!" still matches "แล้ว"
	s = strings.TrimRight(s, "!?ฯ.… ")
	if s == "" {
		return false
	}
	for w := range enders {
		if strings.HasSuffix(s, w) {
			return true
		}
	}
	return false
}

func startsWithStarter(s string) bool {
	for w := range starters {
		if strings.HasPrefix(s, w) {
			return true
		}
	}
	return false
}

// endsWithSentPunct reports a sentence-final punctuation mark: ! ? ฯ or an
// ellipsis (… or "...").
func endsWithSentPunct(s string) bool {
	s = strings.TrimRight(s, " \t")
	return strings.HasSuffix(s, "!") || strings.HasSuffix(s, "?") ||
		strings.HasSuffix(s, "ฯ") || strings.HasSuffix(s, "…") || strings.HasSuffix(s, "...")
}

// Engine is a selectable sentence splitter, so a pipeline can trade speed for
// quality by configuration. Whitespace and Heuristic are pure-Go and need no
// model; the CRF engine (sentence/crf) also satisfies this interface.
type Engine interface {
	Split(text string) []string
}

// EngineFunc adapts a plain function to Engine.
type EngineFunc func(string) []string

// Split calls the underlying function.
func (f EngineFunc) Split(text string) []string { return f(text) }

var (
	// Whitespace splits on every whitespace run — fastest, no model, but
	// over-splits Thai (spaces occur mid-sentence). Matches Split.
	Whitespace Engine = EngineFunc(Split)
	// Heuristic keeps whitespace as split candidates but decides each from
	// ender/starter words and sentence punctuation — near-whitespace speed,
	// far fewer false splits, no model. Matches SplitHeuristic.
	Heuristic Engine = EngineFunc(SplitHeuristic)
)
