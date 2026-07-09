// Package chunk splits text into embedding-sized chunks for RAG ingestion,
// keeping exact byte offsets into the source so every chunk can be cited and
// highlighted: input[Start:End] == Text always holds, including for overlaps.
//
// Splitting is hierarchical, tuned for Thai prose (no sentence-final
// punctuation; long-form prose often puts one paragraph per line):
//
//  1. Paragraphs — the input is cut at every line break ('\n', '\r' or
//     "\r\n"; a single newline is a paragraph break) and each paragraph is
//     trimmed of surrounding whitespace. A paragraph whose Measure fits
//     MaxUnits is kept whole without invoking the sentence splitter.
//  2. Sentences — an oversized paragraph is split by Options.Sentences
//     (default: an offset-preserving whitespace splitter; Thai marks
//     sentence boundaries with spaces) and whole sentences are packed
//     greedily into chunks of at most MaxUnits.
//  3. Hard cuts — only a single sentence that alone exceeds MaxUnits is cut
//     at rune boundaries. A cut is never placed before a Thai combining mark
//     (U+0E31, U+0E34..U+0E3A, U+0E47..U+0E4E), so a mark is never split
//     from its base; when marks leave no legal cut (a base followed by more
//     than MaxUnits marks, or a single rune measuring over MaxUnits) the
//     minimal unsplittable piece is emitted even though it exceeds MaxUnits.
//
// Whitespace rule: chunks never span a line break, and whitespace at either
// edge of a chunk belongs to no chunk — chunk boundaries are trimmed, so the
// gaps between consecutive chunks' non-overlapping parts consist solely of
// whitespace (paragraph separators and inter-sentence spaces at chunk
// edges). Whitespace strictly inside a chunk is preserved as-is.
//
// Overlap: with OverlapUnits > 0, each chunk after the first within a
// paragraph starts with the last whole sentences of the previous chunk
// totalling at most OverlapUnits (never partial sentences; if the previous
// chunk's last sentence alone exceeds OverlapUnits there is no overlap). The
// overlap counts against the new chunk's MaxUnits and is shrunk when needed
// to fit the next sentence. Overlap never crosses a paragraph break, so it
// is always real contiguous source text and [Start,End) ranges of
// consecutive chunks simply overlap.
//
// Measure cost: Measure may be an expensive LLM tokenizer, so it is called
// once per paragraph, once per sentence (each sentence is measured exactly
// once, with the whitespace that follows it inside the paragraph, so
// separators are budgeted), and O(log runes) times per hard-cut piece. A
// chunk's budget is the sum of its sentences' measures: exact for additive
// measures such as the default rune count, and an upper bound for any
// subadditive Measure (real tokenizers). Hard cutting binary-searches the
// longest fitting prefix and therefore assumes Measure is non-decreasing
// over prefixes.
//
// For higher-quality Thai sentence boundaries pass the CRF splitter from
// sentence/crf, whose output concatenates back to its input for the trimmed
// single-line paragraphs this package hands it (verified by tests; Split
// additionally validates every Sentences call and falls back to the
// built-in splitter on any mismatch):
//
//	chunks, err := chunk.Split(text, chunk.Options{
//		MaxUnits:  512,
//		Sentences: crf.Split,
//	})
//
// chunk deliberately does not import sentence/crf so that importers who do
// not need it avoid the embedded model.
package chunk

import (
	"errors"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Chunk is one piece of the input. Text is always the exact source slice:
// input[Start:End] == Text.
type Chunk struct {
	Text       string
	Start, End int
}

// Options configures Split. MaxUnits is required; the zero value of every
// other field is valid.
type Options struct {
	// MaxUnits is the hard cap per chunk, in Measure units. Required > 0.
	// Only the documented unsplittable hard-cut pieces may exceed it.
	MaxUnits int

	// OverlapUnits is the maximum overlap between consecutive chunks of a
	// paragraph, in Measure units (0 = none). Must be < MaxUnits.
	OverlapUnits int

	// Measure returns the size of a piece of text in caller-defined units
	// (e.g. LLM tokens). nil = utf8.RuneCountInString. See the package
	// documentation for how often it is called and the additivity caveat.
	Measure func(string) int

	// Sentences splits one paragraph (a single trimmed line, no line
	// breaks) into sentences. The returned segments must concatenate
	// exactly to the input and start at rune boundaries; otherwise Split
	// falls back to the built-in whitespace splitter for that paragraph.
	// Only invoked for paragraphs whose Measure exceeds MaxUnits.
	// nil = built-in splitter (segments end after each whitespace run,
	// matching the boundaries of package sentence's Split).
	Sentences func(string) []string
}

var (
	ErrMaxUnits = errors.New("chunk: MaxUnits must be > 0")
	ErrOverlap  = errors.New("chunk: OverlapUnits must be >= 0 and less than MaxUnits")
)

// Thai combining marks: a hard cut never lands between one of these and the
// rune before it. U+0E33 (SARA AM) is spacing and deliberately excluded.
const (
	thMaiHanAkat      = 'ั' // MAI HAN-AKAT
	thFirstBelowAbove = 'ิ' // SARA I .. PHINTHU
	thLastBelowAbove  = 'ฺ'
	thFirstToneSign   = '็' // MAITAIKHU .. YAMAKKAN
	thLastToneSign    = '๎'
)

func isThaiMark(r rune) bool {
	return r == thMaiHanAkat ||
		(r >= thFirstBelowAbove && r <= thLastBelowAbove) ||
		(r >= thFirstToneSign && r <= thLastToneSign)
}

const lineBreaks = "\n\r"

// Split chunks text according to o. Empty or whitespace-only input returns
// an empty slice and nil error.
func Split(text string, o Options) ([]Chunk, error) {
	if o.MaxUnits <= 0 {
		return nil, ErrMaxUnits
	}
	if o.OverlapUnits < 0 || o.OverlapUnits >= o.MaxUnits {
		return nil, ErrOverlap
	}
	s := splitter{text: text, o: o, measure: o.Measure}
	if s.measure == nil {
		s.measure = utf8.RuneCountInString
	}
	pos := 0
	for pos < len(text) {
		end := len(text)
		if nl := strings.IndexAny(text[pos:], lineBreaks); nl >= 0 {
			end = pos + nl
		}
		if ps, pe := trimRange(text, pos, end); ps < pe {
			s.paragraph(ps, pe)
		}
		pos = end + 1
	}
	return s.out, nil
}

// unit is one packable piece: a whole sentence (sent), or a hard-cut slice
// of an oversized sentence. m caches Measure(text[start:end]).
type unit struct {
	start, end int
	m          int
	sent       bool
}

type splitter struct {
	text    string
	o       Options
	measure func(string) int
	units   []unit // current paragraph's units (reused)
	cuts    []int  // sentence end offsets within the paragraph (reused)
	bounds  []int  // rune boundaries of the sentence being hard-cut (reused)
	out     []Chunk
}

func (s *splitter) paragraph(ps, pe int) {
	par := s.text[ps:pe]
	if s.measure(par) <= s.o.MaxUnits {
		s.emit(ps, pe)
		return
	}
	s.units = s.units[:0]
	s.buildUnits(ps, par)
	s.pack()
}

func (s *splitter) buildUnits(ps int, par string) {
	if s.o.Sentences == nil || !s.customCuts(par) {
		s.cuts = builtinCuts(par, s.cuts[:0])
	}
	prev := 0
	for _, cut := range s.cuts {
		seg := par[prev:cut]
		if m := s.measure(seg); m <= s.o.MaxUnits {
			s.units = append(s.units, unit{ps + prev, ps + cut, m, true})
		} else {
			s.hardCut(ps+prev, seg)
		}
		prev = cut
	}
}

// builtinCuts appends the end offset of each built-in sentence: a sentence
// runs up to (and includes) the whitespace run that follows it, so segments
// tile par exactly and their boundaries match package sentence's Split.
func builtinCuts(par string, cuts []int) []int {
	inSpace := false
	for i, r := range par {
		switch {
		case unicode.IsSpace(r):
			inSpace = true
		case inSpace:
			cuts = append(cuts, i)
			inSpace = false
		}
	}
	return append(cuts, len(par))
}

// customCuts converts o.Sentences(par) into end offsets, validating that the
// segments concatenate exactly to par and start at rune boundaries. Reports
// whether the output was valid; on false the caller falls back to
// builtinCuts.
func (s *splitter) customCuts(par string) bool {
	segs := s.o.Sentences(par)
	s.cuts = s.cuts[:0]
	off := 0
	for _, seg := range segs {
		if seg == "" {
			continue
		}
		end := off + len(seg)
		if end > len(par) || par[off:end] != seg {
			return false
		}
		if end < len(par) && !utf8.RuneStart(par[end]) {
			return false
		}
		s.cuts = append(s.cuts, end)
		off = end
	}
	return off == len(par)
}

// hardCut splits an oversized sentence at rune boundaries into pieces each
// measuring at most MaxUnits (binary search for the longest fitting prefix,
// assuming Measure is non-decreasing over prefixes), shifting every cut off
// Thai combining marks. See the package documentation for the unsplittable
// exceptions.
func (s *splitter) hardCut(base int, seg string) {
	s.bounds = s.bounds[:0]
	for i := range seg {
		s.bounds = append(s.bounds, i)
	}
	s.bounds = append(s.bounds, len(seg))
	b := s.bounds
	last := len(b) - 1
	for pos := 0; pos < last; {
		lo, hi := pos+1, last
		for lo < hi {
			mid := lo + (hi-lo+1)/2
			if s.measure(seg[b[pos]:b[mid]]) <= s.o.MaxUnits {
				lo = mid
			} else {
				hi = mid - 1
			}
		}
		cut := s.avoidMarkSplit(seg, lo, pos, last)
		piece := seg[b[pos]:b[cut]]
		s.units = append(s.units, unit{base + b[pos], base + b[cut], s.measure(piece), false})
		pos = cut
	}
}

// avoidMarkSplit moves a candidate cut (an index into s.bounds) off any Thai
// combining mark: first retreating (piece stays within budget), then, when
// the marks reach back to the minimal one-rune piece, extending past them.
func (s *splitter) avoidMarkSplit(seg string, cut, pos, last int) int {
	b := s.bounds
	for cut < last && cut > pos+1 && isMarkAt(seg, b[cut]) {
		cut--
	}
	for cut < last && isMarkAt(seg, b[cut]) {
		cut++
	}
	return cut
}

func isMarkAt(seg string, off int) bool {
	r, _ := utf8.DecodeRuneInString(seg[off:])
	return isThaiMark(r)
}

// pack greedily fills chunks with the current paragraph's units, seeding
// each chunk after the first with the overlap suffix of the previous one.
func (s *splitter) pack() {
	u := s.units
	prevFirst, prevStart := 0, -1
	for i := 0; i < len(u); {
		first, sum := i, 0
		if i > 0 && s.o.OverlapUnits > 0 {
			first, sum = s.overlapStart(prevFirst, i, prevStart)
		}
		k := i
		for k < len(u) && sum+u[k].m <= s.o.MaxUnits {
			sum += u[k].m
			k++
		}
		if k == i {
			k = i + 1 // single unsplittable unit over MaxUnits
		}
		if start, ok := s.emit(u[first].start, u[k-1].end); ok {
			prevStart = start
		}
		prevFirst, i = first, k
	}
}

// overlapStart picks the overlap for the chunk whose first new unit is i:
// the longest suffix of whole sentences of the previous chunk (units
// [prevFirst,i)) whose measures total at most OverlapUnits, then shrunk from
// the front until the first new unit fits MaxUnits and the chunk starts
// strictly after the previously emitted chunk (so no chunk is ever contained
// in its successor — possible otherwise when boundary whitespace trimming
// leaves the previous chunk with only its final sentence visible).
func (s *splitter) overlapStart(prevFirst, i, prevStart int) (first, sum int) {
	u := s.units
	j := i
	for j > prevFirst && u[j-1].sent && sum+u[j-1].m <= s.o.OverlapUnits {
		sum += u[j-1].m
		j--
	}
	for j < i && (u[j].start <= prevStart || sum+u[i].m > s.o.MaxUnits) {
		sum -= u[j].m
		j++
	}
	return j, sum
}

// emit appends text[start:end] as a chunk, trimming boundary whitespace
// (the documented whitespace rule); whitespace-only spans are dropped.
func (s *splitter) emit(start, end int) (int, bool) {
	start, end = trimRange(s.text, start, end)
	if start >= end {
		return start, false
	}
	s.out = append(s.out, Chunk{Text: s.text[start:end], Start: start, End: end})
	return start, true
}

func trimRange(text string, start, end int) (int, int) {
	for start < end {
		r, n := utf8.DecodeRuneInString(text[start:end])
		if !unicode.IsSpace(r) {
			break
		}
		start += n
	}
	for end > start {
		r, n := utf8.DecodeLastRuneInString(text[start:end])
		if !unicode.IsSpace(r) {
			break
		}
		end -= n
	}
	return start, end
}
