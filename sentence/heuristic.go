package sentence

import (
	_ "embed"
	"strings"
	"unicode/utf8"
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

// SplitHeuristic splits text into sentences with a rule engine informed by the
// Thai sentence-segmentation literature (Slayden et al. 2010; Charoenpornsawat
// 2001): whitespace runs are the only split candidates, and each is decided
// from cheap signals — ender/starter words, sentence punctuation, and the
// non-break suppressors that stop the classic Thai over-splitting:
//
//   - inside a bracket/quote pair — a space there is never a sentence break;
//   - right after an honorific (นาย/ดร./ด.ช. …) — the name's internal spaces
//     are prescriptive, not sentence breaks;
//   - between a number and a month (or a month and a year) — a date's spaces.
//
// No tokenizer and no model, so it stays near whitespace speed while cutting the
// false splits raw whitespace makes.
func SplitHeuristic(text string) []string { return splitHeur(text, defaultMaxRun) }

// defaultMaxRun: a space is also cut when the current sentence has grown past
// this many runes without a lexical cue (Slayden distance feature). Default 0
// boundaries between plain content words. Tuned on UD_Thai-PUD (see eval).
const defaultMaxRun = 0

func splitHeur(text string, maxRun int) []string {
	pieces, hardBreak := splitKeepingBreaks(text)
	if len(pieces) == 0 {
		return nil
	}
	var out []string
	cur := pieces[0]
	depth := bracketDelta(pieces[0])
	honorWin := 0
	if isHonorific(pieces[0]) {
		honorWin = honorWindow
	}
	for i := 1; i < len(pieces); i++ {
		next := pieces[i]
		var boundary bool
		switch {
		case hardBreak[i]:
			boundary = true // newline: hard break, overrides suppressors
		case depth > 0 || honorWin > 0 || isDateGap(cur, next):
			boundary = false // suppressed: inside brackets / after honorific / a date
		default:
			boundary = endsWithSentPunct(cur) || endsWithEnder(cur) || startsWithStarter(next) ||
				(maxRun > 0 && utf8.RuneCountInString(cur) >= maxRun) // distance feature
		}
		if boundary {
			if s := strings.TrimSpace(cur); s != "" {
				out = append(out, s)
			}
			cur = next
		} else {
			cur = cur + " " + next
		}
		depth += bracketDelta(next)
		if depth < 0 {
			depth = 0
		}
		if honorWin > 0 {
			honorWin--
		}
		if isHonorific(next) {
			honorWin = honorWindow
		}
	}
	if s := strings.TrimSpace(cur); s != "" {
		out = append(out, s)
	}
	return out
}

const honorWindow = 2 // segments after an honorific whose spaces are non-breaks

// honorifics: a following name's internal spaces are not sentence breaks.
var honorifics = map[string]bool{
	"นาย": true, "นาง": true, "นางสาว": true, "น.ส.": true, "ด.ช.": true, "ด.ญ.": true,
	"เด็กชาย": true, "เด็กหญิง": true, "ดร.": true, "ศ.": true, "รศ.": true, "ผศ.": true,
	"คุณ": true, "พล.": true, "พ.ต.": true, "ร.ต.": true, "พระ": true, "หม่อม": true,
}

func isHonorific(s string) bool {
	s = strings.TrimSpace(s)
	if honorifics[s] {
		return true
	}
	for h := range honorifics {
		if strings.HasPrefix(s, h) { // "นายสมชาย" written joined
			return true
		}
	}
	return false
}

var thaiMonths = map[string]bool{
	"มกราคม": true, "กุมภาพันธ์": true, "มีนาคม": true, "เมษายน": true, "พฤษภาคม": true,
	"มิถุนายน": true, "กรกฎาคม": true, "สิงหาคม": true, "กันยายน": true, "ตุลาคม": true,
	"พฤศจิกายน": true, "ธันวาคม": true,
	"ม.ค.": true, "ก.พ.": true, "มี.ค.": true, "เม.ย.": true, "พ.ค.": true, "มิ.ย.": true,
	"ก.ค.": true, "ส.ค.": true, "ก.ย.": true, "ต.ค.": true, "พ.ย.": true, "ธ.ค.": true,
}

// isDateGap reports a space that sits inside a date: a number then a month, or a
// month then a number (the year) — "1 มกราคม 2567".
func isDateGap(prev, next string) bool {
	prev, next = strings.TrimSpace(prev), strings.TrimSpace(next)
	if endsWithDigit(prev) && startsWithMonth(next) {
		return true
	}
	if isMonth(prev) && startsWithDigit(next) {
		return true
	}
	return false
}

func isMonth(s string) bool { return thaiMonths[strings.TrimSpace(s)] }

func startsWithMonth(s string) bool {
	for m := range thaiMonths {
		if strings.HasPrefix(s, m) {
			return true
		}
	}
	return false
}

func endsWithDigit(s string) bool {
	r := []rune(s)
	if len(r) == 0 {
		return false
	}
	return isDigitRune(r[len(r)-1])
}

func startsWithDigit(s string) bool {
	r := []rune(s)
	return len(r) > 0 && isDigitRune(r[0])
}

func isDigitRune(r rune) bool {
	return (r >= '0' && r <= '9') || (r >= '๐' && r <= '๙') // ASCII or Thai digits
}

// bracketDelta returns the net bracket/quote nesting change in s (opens minus
// closes), for the paired marks common in Thai text.
func bracketDelta(s string) int {
	d := 0
	for _, r := range s {
		switch r {
		case '(', '[', '{', '（', '「', '“', '‘', '«', '『', '〈':
			d++
		case ')', ']', '}', '）', '」', '”', '’', '»', '』', '〉':
			d--
		}
	}
	return d
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
	if strings.HasSuffix(s, "ๆ") { // maiyamok repetition mark is a sentence-end cue
		return true
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

// endsWithSentPunct reports a sentence-final punctuation mark: ! ? … "..." or
// the archaic Thai terminators ๚ ๛. NOT ฯ (paiyannoi) — that is an abbreviation
// mark (กรุงเทพฯ) and ฯลฯ means "etc." mid-list, so it never ends a sentence.
func endsWithSentPunct(s string) bool {
	s = strings.TrimRight(s, " \t")
	return strings.HasSuffix(s, "!") || strings.HasSuffix(s, "?") ||
		strings.HasSuffix(s, "…") || strings.HasSuffix(s, "...") ||
		strings.HasSuffix(s, "๚") || strings.HasSuffix(s, "๛")
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

// SplitHeuristicTuned exposes the distance parameter for benchmarking/tuning
// (maxRun runes; 0 disables the distance cut).
func SplitHeuristicTuned(text string, maxRun int) []string { return splitHeur(text, maxRun) }
