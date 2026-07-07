// Package charseg is an "outside-the-box" fast Thai sentence segmenter: a
// CHARACTER-LEVEL, single-pass, NO-TOKENIZER boundary classifier.
//
// Motivation. The CRF segmenter (sentence/crf) spends ~33% of its time on
// tokenization and ~67% on hand-crafted word-n-gram feature extraction; the
// 2-state Viterbi itself is trivial. So the cost is TOKENIZE + FEATURE
// ENGINEERING, not "ML". This package removes both stages: it classifies
// boundaries directly from raw characters in one pass, with no word tokenizer.
//
// Design. Thai sentence boundaries in practice fall at a small set of
// CANDIDATE positions — before whitespace, after terminal punctuation, or
// before an opening dialogue quote — each detectable in O(1) from the
// characters alone (no tokenizer). At every candidate the classifier scores
// character n-grams (1..3) in a ±3-rune window plus cheap character-type
// features (space / digit / Thai / Latin / punct / quote / script-change) and
// predicts boundary-after (binary). Features are hashed into a fixed weight
// vector (the "hashing trick"), so inference does array indexing only — no map
// lookups, no per-position allocation. Weights are fit with an averaged
// perceptron (dependency-free; same family as crf.Train).
//
// This mirrors the sentence.Heuristic candidate set (whitespace/punct) but
// replaces its hand-written ender/starter word lists with a learned
// character-n-gram model — closing the accuracy gap to the CRF while keeping
// near-heuristic speed and needing no tokenizer.
package charseg

import (
	"bufio"
	_ "embed"
	"io"
	"sort"
	"strconv"
	"strings"
)

// Model is a hashed averaged-perceptron boundary classifier. After loading it
// is read-only and safe for concurrent use.
type Model struct {
	w    []float32 // hashed weight vector; len is a power of two
	mask uint64
}

const fnvOff = 1469598103934665603
const fnvPrime = 1099511628211

func hstart(kind uint64) uint64 { h := uint64(fnvOff); h ^= kind; h *= fnvPrime; return h }
func hstep(h, v uint64) uint64  { h ^= v; h *= fnvPrime; return h }

// pad is a rune sentinel for out-of-range window positions (distinct from any
// real rune, which is <= 0x10FFFF).
const pad uint64 = 1 << 32

func rn(r []rune, idx int) uint64 {
	if idx < 0 || idx >= len(r) {
		return pad
	}
	return uint64(r[idx])
}

// character type classes (used as features and for script-change detection).
const (
	tOther uint64 = iota
	tSpace
	tDigit
	tThai
	tLatin
	tPunct
	tQuote
	tPad
)

func runeType(v uint64) uint64 {
	if v == pad {
		return tPad
	}
	r := rune(v)
	switch {
	case r == ' ' || r == '\t' || r == '\n' || r == '\r' || r == '\v' || r == '\f' || r == 0x00A0:
		return tSpace
	case (r >= '0' && r <= '9') || (r >= '๐' && r <= '๙'):
		return tDigit
	case r >= 0x0E00 && r <= 0x0E7F:
		return tThai
	case (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= 0x00C0 && r <= 0x024F):
		return tLatin
	}
	switch r {
	case '“', '”', '"', '‘', '’', '\'', '«', '»', '「', '」', '『', '』', '„', '〈', '〉':
		return tQuote
	case '.', ',', '!', '?', ';', ':', '…', '。', '！', '？', '、', '·', '(', ')', '[', ']', '{', '}', '-':
		return tPunct
	}
	return tOther
}

// isThaiConsonant reports a Thai consonant rune (ก..ฮ), used by the
// abbreviation-pattern feature (short Thai cluster + period, e.g. พ.ศ., ด.ช.).
func isThaiConsonant(r rune) bool { return r >= 0x0E01 && r <= 0x0E2E }

// Sentence-final particle cues (token-final signals that a candidate space is a
// sentence boundary). Curated by register — modern colloquial, question,
// emotive, classical/royal-court, Chinese- and Japanese-translated web-novel,
// poetic, and archaic-formal — covering both the eval registers and the
// production novel domains. These are END-of-sentence signals only; forms of
// address / pronouns that open or sit mid-sentence are excluded (see the data
// file header). A token whose stripped form ENDS WITH one of these fires a
// hashed feature whose weight the perceptron LEARNS (not a hard cut), so
// overlaps and sometimes-medial items are safe. The list is embedded from
// data/final_particles.txt and sorted longest-first so the suffix scan prefers
// the most specific match (เจ้าค่ะ before ค่ะ).
//
//go:embed data/final_particles.txt
var finalParticlesData string

var finalParticles = loadRuneList(finalParticlesData)

// Sentence-opener cues: tokens that commonly begin a new sentence right after a
// candidate space (a weak boundary signal on the following side).
var sentenceOpeners = [][]rune{
	[]rune("อย่างไรก็ตาม"), []rune("ดังนั้น"), []rune("เมื่อ"), []rune("ซึ่ง"),
	[]rune("โดย"), []rune("แต่"), []rune("และ"),
}

// loadRuneList parses a whitespace/newline-separated word list (with '#'
// comment lines) into deduplicated rune slices, sorted longest-first so a
// suffix/prefix scan matches the most specific entry.
func loadRuneList(data string) [][]rune {
	seen := map[string]bool{}
	var words []string
	for _, line := range strings.Split(data, "\n") {
		if s := strings.TrimSpace(line); s == "" || strings.HasPrefix(s, "#") {
			continue
		}
		for _, w := range strings.Fields(line) {
			if w == "" || strings.HasPrefix(w, "#") || seen[w] {
				continue
			}
			seen[w] = true
			words = append(words, w)
		}
	}
	sort.SliceStable(words, func(i, j int) bool {
		return len([]rune(words[i])) > len([]rune(words[j]))
	})
	out := make([][]rune, len(words))
	for i, w := range words {
		out[i] = []rune(w)
	}
	return out
}

// matchSuffix reports whether the runes ending at index i (inclusive) equal p.
func matchSuffix(r []rune, i int, p []rune) bool {
	start := i - len(p) + 1
	if start < 0 {
		return false
	}
	for k := 0; k < len(p); k++ {
		if r[start+k] != p[k] {
			return false
		}
	}
	return true
}

// matchPrefix reports whether the runes starting at index j equal p.
func matchPrefix(r []rune, j int, p []rune) bool {
	if j < 0 || j+len(p) > len(r) {
		return false
	}
	for k := 0; k < len(p); k++ {
		if r[j+k] != p[k] {
			return false
		}
	}
	return true
}

// firstNonSpaceAfter returns the index of the first non-whitespace rune strictly
// after index i, or -1 if none.
func firstNonSpaceAfter(r []rune, i int) int {
	for j := i + 1; j < len(r); j++ {
		if !isSpaceRune(r[j]) {
			return j
		}
	}
	return -1
}

// prevTokenLen returns the length (in runes, capped) of the non-space run that
// ends at index i — the "preceding token" before a candidate space.
func prevTokenLen(r []rune, i int) int {
	n := 0
	for j := i; j >= 0 && !isSpaceRune(r[j]) && n < 64; j-- {
		n++
	}
	return n
}

const maxFeat = 64

// Ablation toggles for the discriminative boundary features (all default on).
// featParticles: sentence-final particle cue (group 7). featOpener: sentence-
// opener cue (group 8). featShape: char-class pair / abbreviation-dot / token-
// length shape features (groups 9-11).
const (
	featParticles = true
	featOpener    = true
	featShape     = true
)

// features writes the (unmasked) hashed feature ids for a boundary decision
// after rune i into out and returns the count. Character n-grams (1..3) over a
// ±3-rune window plus character-type features. Allocation-free (fixed array).
func features(r []rune, i int, out *[maxFeat]uint64) int {
	c0 := rn(r, i-3)
	c1 := rn(r, i-2)
	c2 := rn(r, i-1)
	c3 := rn(r, i)
	c4 := rn(r, i+1)
	c5 := rn(r, i+2)
	c6 := rn(r, i+3)
	c := [7]uint64{c0, c1, c2, c3, c4, c5, c6}

	n := 0
	out[n] = hstart(0) // bias
	n++

	// unigrams (7)
	for k := 0; k < 7; k++ {
		h := hstart(1)
		h = hstep(h, uint64(k))
		h = hstep(h, c[k])
		out[n] = h
		n++
	}
	// bigrams (6)
	for k := 0; k < 6; k++ {
		h := hstart(2)
		h = hstep(h, uint64(k))
		h = hstep(h, c[k])
		h = hstep(h, c[k+1])
		out[n] = h
		n++
	}
	// trigrams (5)
	for k := 0; k < 5; k++ {
		h := hstart(3)
		h = hstep(h, uint64(k))
		h = hstep(h, c[k])
		h = hstep(h, c[k+1])
		h = hstep(h, c[k+2])
		out[n] = h
		n++
	}
	// type unigrams at offsets -1,0,+1 (3)
	tm1, t0, tp1 := runeType(c2), runeType(c3), runeType(c4)
	for k, tv := range [3]uint64{tm1, t0, tp1} {
		h := hstart(4)
		h = hstep(h, uint64(k))
		h = hstep(h, tv)
		out[n] = h
		n++
	}
	// type bigram (type@0,type@+1) (1)
	h := hstart(5)
	h = hstep(h, t0)
	h = hstep(h, tp1)
	out[n] = h
	n++
	// script-change flag between rune i and i+1 (1)
	sc := uint64(0)
	if t0 != tp1 {
		sc = 1
	}
	out[n] = hstep(hstart(6), sc)
	n++

	// --- discriminative boundary-vs-phrase-internal features (additive) ---
	// These distinguish a sentence-ending space from a phrase-internal space
	// on low-prior/formal registers, without touching the char n-grams above.
	// The group toggles below are for ablation; all default on.

	// (7) sentence-final particle ending the token just before the candidate.
	// Longest match only (the table is sorted longest-first), so a single, most-
	// specific cue fires. A strong end-of-sentence signal (learned weight).
	if featParticles {
		for pid := 0; pid < len(finalParticles); pid++ {
			if matchSuffix(r, i, finalParticles[pid]) {
				out[n] = hstep(hstart(7), uint64(pid+1))
				n++
				break
			}
		}
	}

	if j := firstNonSpaceAfter(r, i); j >= 0 {
		// (8) sentence-opener beginning the next token after the candidate space.
		if featOpener {
			for oid := 0; oid < len(sentenceOpeners); oid++ {
				if matchPrefix(r, j, sentenceOpeners[oid]) {
					out[n] = hstep(hstart(8), uint64(oid+1))
					n++
					break
				}
			}
		}
		// (9) char-class pair across the space: last rune of the previous token
		// vs first rune of the next token (skips the whitespace itself, unlike
		// the type bigram above which straddles rune i / i+1=space).
		if featShape {
			h9 := hstart(9)
			h9 = hstep(h9, runeType(uint64(r[i])))
			h9 = hstep(h9, runeType(uint64(r[j])))
			out[n] = h9
			n++
		}
	}

	// (10) abbreviation-dot pattern (short Thai cluster + period, e.g. พ.ศ.,
	// ด.ช., น.ส.) — a cue to SUPPRESS a false cut after an internal period.
	if featShape && r[i] == '.' && i >= 1 && isThaiConsonant(r[i-1]) &&
		(i < 2 || r[i-2] == '.' || isSpaceRune(r[i-2])) {
		out[n] = hstep(hstart(10), 1)
		n++
	}

	// (11) preceding-token length bucket (very short tokens before a space are
	// less often a sentence end; helps suppress spurious cuts).
	if featShape {
		pl := prevTokenLen(r, i)
		var lb uint64
		switch {
		case pl <= 1:
			lb = 1
		case pl == 2:
			lb = 2
		case pl == 3:
			lb = 3
		case pl <= 5:
			lb = 4
		case pl <= 8:
			lb = 5
		default:
			lb = 6
		}
		out[n] = hstep(hstart(11), lb)
		n++
	}
	return n
}

// scoreAt returns the classifier score for a boundary after rune i.
func (m *Model) scoreAt(r []rune, i int) float64 {
	var buf [maxFeat]uint64
	n := features(r, i, &buf)
	var s float64
	for k := 0; k < n; k++ {
		s += float64(m.w[buf[k]&m.mask])
	}
	return s
}

// ScoreAt returns the raw per-candidate boundary score after rune i (the model
// predicts a boundary at a candidate iff this score exceeds the operating
// threshold τ; the default Boundaries uses τ=0). Exposed for operating-point
// tuning and analysis. The score at non-candidate positions is meaningless
// (those positions are never cut); callers should gate on IsCandidate.
func (m *Model) ScoreAt(r []rune, i int) float64 { return m.scoreAt(r, i) }

// IsCandidate reports whether a boundary is allowed after rune i (see the
// unexported isCandidate). Exposed for operating-point analysis.
func IsCandidate(r []rune, i int) bool { return isCandidate(r, i) }

// isSpaceRune reports an ASCII/NBSP whitespace rune.
func isSpaceRune(r rune) bool {
	return r == ' ' || r == '\t' || r == '\n' || r == '\r' || r == '\v' || r == '\f' || r == 0x00A0
}

func isTerm(r rune) bool {
	switch r {
	case '.', '!', '?', '…', '。', '！', '？':
		return true
	}
	return false
}

func isOpenQuote(r rune) bool {
	switch r {
	case '“', '«', '「', '『', '„':
		return true
	}
	return false
}

// isCandidate reports whether a sentence boundary is allowed after rune i — the
// only positions the classifier evaluates. Detected from characters alone (no
// tokenizer): the doc end, before whitespace, after terminal punctuation, or
// before an opening dialogue quote.
func isCandidate(r []rune, i int) bool {
	if i == len(r)-1 {
		return true
	}
	if isSpaceRune(r[i+1]) {
		return true
	}
	if isTerm(r[i]) {
		return true
	}
	if isOpenQuote(r[i+1]) {
		return true
	}
	return false
}

// Boundaries returns, for the rune slice, whether a sentence boundary follows
// each rune (true only at candidate positions the model cut). Exposed for
// evaluation; the last rune is always a boundary.
func (m *Model) Boundaries(r []rune) []bool {
	b := make([]bool, len(r))
	if len(r) == 0 {
		return b
	}
	for i := 0; i < len(r); i++ {
		if i == len(r)-1 {
			b[i] = true
			continue
		}
		if isCandidate(r, i) && m.scoreAt(r, i) > 0 {
			b[i] = true
		}
	}
	return b
}

// BoundariesTau is like Boundaries but predicts a boundary at a candidate when
// the model score exceeds a caller-supplied threshold τ instead of 0. Lowering
// τ (τ<0) keeps more candidates as boundaries (recall-lean, toward whitespace);
// raising τ is more selective (higher precision). BoundariesTau(r, 0) is
// identical to Boundaries(r). The final rune is always a boundary.
func (m *Model) BoundariesTau(r []rune, tau float64) []bool {
	b := make([]bool, len(r))
	if len(r) == 0 {
		return b
	}
	for i := 0; i < len(r); i++ {
		if i == len(r)-1 {
			b[i] = true
			continue
		}
		if isCandidate(r, i) && m.scoreAt(r, i) > tau {
			b[i] = true
		}
	}
	return b
}

// SplitTau segments text like Split but at operating threshold τ (see
// BoundariesTau). SplitTau(text, 0) is identical to Split(text).
func (m *Model) SplitTau(text string, tau float64) []string {
	if text == "" {
		return nil
	}
	r := []rune(text)
	var out []string
	prev := 0
	for i := 0; i < len(r); i++ {
		cut := i == len(r)-1
		if !cut && isCandidate(r, i) && m.scoreAt(r, i) > tau {
			cut = true
		}
		if cut {
			if s := strings.TrimSpace(string(r[prev : i+1])); s != "" {
				out = append(out, s)
			}
			prev = i + 1
		}
	}
	return out
}

// Split segments text into sentences. Single pass over the runes, no tokenizer.
// Returns nil for empty input.
func (m *Model) Split(text string) []string {
	if text == "" {
		return nil
	}
	r := []rune(text)
	var out []string
	prev := 0
	for i := 0; i < len(r); i++ {
		cut := i == len(r)-1
		if !cut && isCandidate(r, i) && m.scoreAt(r, i) > 0 {
			cut = true
		}
		if cut {
			if s := strings.TrimSpace(string(r[prev : i+1])); s != "" {
				out = append(out, s)
			}
			prev = i + 1
		}
	}
	return out
}

// Split segments text with the default (permissive, license-clean) model.
func Split(text string) []string { return Default().Split(text) }

// RecallLeanTau is the operating point (τ) that maximizes macro boundary-F1 for
// the embedded Default model across a 10-register Thai eval: leave-one-dataset-out
// honest macro boundary-F1 ≈ 0.698 vs 0.663 at τ=0 (+0.035), reaching ~96% of the
// best-per-register oracle. It keeps more candidate spaces as boundaries (higher
// recall, still well above whitespace on precision). It is MODEL-SPECIFIC — the
// value is tied to this model's score scale; retraining requires re-tuning via a
// τ sweep. For precision-sensitive use keep the τ=0 default (Split).
const RecallLeanTau = -11.0

// RecallLean segments text with the default model at the recall-leaning operating
// point (RecallLeanTau): it trades a little precision for recall and lifts macro
// boundary-F1 ~+0.035 over Split on a fair multi-domain eval. Equivalent to
// Default().SplitTau(text, RecallLeanTau).
func RecallLean(text string) []string { return Default().SplitTau(text, RecallLeanTau) }

// NewModel builds a Model from a hashed weight slice (len must be a power of
// two). Used by the trainer.
func NewModel(w []float32) *Model {
	return &Model{w: w, mask: uint64(len(w) - 1)}
}

// Save writes the model as: a "#bits\tN" header then one "index\tweight" line
// per nonzero weight (sparse).
func (m *Model) Save(wtr io.Writer) error {
	bw := bufio.NewWriter(wtr)
	bits := 0
	for (1 << bits) < len(m.w) {
		bits++
	}
	if _, err := bw.WriteString("#bits\t" + strconv.Itoa(bits) + "\n"); err != nil {
		return err
	}
	for i, v := range m.w {
		if v == 0 {
			continue
		}
		if _, err := bw.WriteString(strconv.Itoa(i)); err != nil {
			return err
		}
		if err := bw.WriteByte('\t'); err != nil {
			return err
		}
		if _, err := bw.WriteString(strconv.FormatFloat(float64(v), 'g', -1, 32)); err != nil {
			return err
		}
		if err := bw.WriteByte('\n'); err != nil {
			return err
		}
	}
	return bw.Flush()
}

// LoadModel reads a model written by Save.
func LoadModel(rdr io.Reader) (*Model, error) {
	sc := bufio.NewScanner(rdr)
	sc.Buffer(make([]byte, 0, 1<<16), 1<<20)
	bits := 20
	var w []float32
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "#bits\t") {
			bits, _ = strconv.Atoi(strings.TrimSpace(line[len("#bits\t"):]))
			w = make([]float32, 1<<bits)
			continue
		}
		if w == nil {
			w = make([]float32, 1<<bits)
		}
		t := strings.IndexByte(line, '\t')
		if t < 0 {
			continue
		}
		idx, err := strconv.Atoi(line[:t])
		if err != nil || idx < 0 || idx >= len(w) {
			continue
		}
		v, _ := strconv.ParseFloat(line[t+1:], 32)
		w[idx] = float32(v)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if w == nil {
		w = make([]float32, 1<<bits)
	}
	return NewModel(w), nil
}
