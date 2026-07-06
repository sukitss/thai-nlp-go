package main

// Novel-domain weak supervision.
//
// We have no gold sentence-boundary corpus for the novel domain (translated
// Chinese/Japanese + Thai web novels). Instead we derive a *silver* corpus from
// the orthography of novel text we own (the TNV1 platform): dialogue quotation
// marks and terminal punctuation are high-precision sentence-boundary signals,
// whereas bare spaces in this domain are mostly intra-sentence clause separators
// (the very ambiguity crfcut is meant to resolve). See -novel in main.go.
//
// The labelling is purely orthographic (quotes + terminal punctuation), so it is
// independent of the CRF's own lexical features (word / ender / starter n-grams);
// the model must learn from context which tokens precede a real boundary. This
// keeps the before/after comparison honest rather than the CRF trivially
// re-deriving the labelling rule.

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"os"
	"strings"

	"github.com/sukitss/thai-nlp-go/sentence/crf"
	"github.com/sukitss/thai-nlp-go/tokenize"
)

// closingSuffixes end a sentence when a token ends with one of them: closing
// quotation marks (Chinese/Japanese/guillemets/typographic/ASCII) and Thai/Latin
// terminal punctuation.
var closingSuffixes = []string{
	"”", "\"", "»", "」", "』", "’", // closing quotes
	".", "!", "?", "…", "。", "！", "？", // terminal punctuation
}

// openingPrefixes start a new dialogue turn: a token beginning with one of these
// implies the previous sentence ended just before it.
var openingPrefixes = []string{
	"“", "«", "「", "『", "„", // opening quotes
}

func endsSentence(tok string) bool {
	for _, s := range closingSuffixes {
		if strings.HasSuffix(tok, s) {
			return true
		}
	}
	return false
}

func startsDialogue(tok string) bool {
	for _, p := range openingPrefixes {
		if strings.HasPrefix(tok, p) {
			return true
		}
	}
	return false
}

// silverLabels assigns I/E labels to a token stream (whitespace kept, as from
// tokenize.Segment) using the orthographic boundary heuristic.
func silverLabels(toks []string) []byte {
	labs := make([]byte, len(toks))
	for i := range labs {
		labs[i] = 'I'
	}
	lastContent := -1
	for i, t := range toks {
		st := strings.TrimSpace(t)
		if st == "" {
			continue
		}
		if startsDialogue(st) && lastContent >= 0 {
			labs[lastContent] = 'E' // cut before a new dialogue turn
		}
		if endsSentence(st) {
			labs[i] = 'E' // cut after a closing quote / terminal punctuation
		}
		lastContent = i
	}
	if lastContent >= 0 {
		labs[lastContent] = 'E' // always end the document
	}
	return labs
}

// countE returns the number of 'E' (boundary) labels.
func countE(labs []byte) int {
	n := 0
	for _, b := range labs {
		if b == 'E' {
			n++
		}
	}
	return n
}

// runNovel builds a silver corpus from novel CSV(s), splits it deterministically
// into train / held-out, trains a CRF, and prints a before/after comparison
// (whitespace baseline, embedded crfcut, retrained) on the held-out split.
func readNovelSet(csvArg, col string) []crf.Example {
	var all []crf.Example
	for _, p := range strings.Split(csvArg, ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		docs := readNovelCSV(p, col)
		fmt.Printf("novel: %d docs from %s\n", len(docs), p)
		all = append(all, docs...)
	}
	return all
}

func statE(docs []crf.Example) (tok, e int) {
	for _, ex := range docs {
		tok += len(ex.Labels)
		e += countE(ex.Labels)
	}
	return
}

func runNovel(csvArg, evalArg, col string, holdout float64, iters int, out string) {
	all := readNovelSet(csvArg, col)
	if len(all) == 0 {
		fmt.Fprintln(os.Stderr, "no novel documents")
		os.Exit(1)
	}

	var train, test []crf.Example
	if evalArg != "" {
		// Separate held-out file(s): cleanest generalization test (disjoint text).
		train = all
		test = readNovelSet(evalArg, col)
	} else if holdout <= 0 {
		train = all // train on everything (final shippable model; no eval)
	} else {
		// Deterministic held-out split: every k-th document (by stride) is held
		// out, so train/test are disjoint and reproducible without shuffling.
		stride := int(1.0 / holdout)
		if stride < 2 {
			stride = 2
		}
		for i, ex := range all {
			if i%stride == 0 {
				test = append(test, ex)
			} else {
				train = append(train, ex)
			}
		}
	}
	trTok, trE := statE(train)
	teTok, teE := statE(test)
	fmt.Printf("split: train=%d docs (%d tok, %d E) held-out=%d docs (%d tok, %d E)\n",
		len(train), trTok, trE, len(test), teTok, teE)

	model := crf.Train(train, iters)

	if out != "" {
		f, err := os.Create(out)
		must(err)
		must(model.Save(f))
		f.Close()
		fmt.Printf("saved model -> %s\n", out)
	}

	if len(test) == 0 {
		return
	}
	fmt.Printf("\n=== held-out eval (silver, %d docs) ===\n", len(test))
	fmt.Printf("  whitespace  : %s\n", crf.Eval(crf.WhitespaceBaseline, test))
	fmt.Printf("  crfcut(TED) : %s\n", crf.Eval(crf.Default().Labels, test))
	fmt.Printf("  trained     : %s\n", crf.Eval(model.Labels, test))
}

// readNovelCSV reads one CSV (with a header) and returns one silver Example per
// row, using the given text column (default: "text", else the first column).
// Rows are tokenized with the default segmenter (whitespace kept) and labelled
// by silverLabels. Empty / boundary-less rows are skipped.
func readNovelCSV(path, col string) []crf.Example {
	f, err := os.Open(path)
	must(err)
	defer f.Close()

	r := csv.NewReader(bufio.NewReader(f))
	r.FieldsPerRecord = -1
	r.LazyQuotes = true

	header, err := r.Read()
	must(err)
	textIdx := 0
	for i, h := range header {
		if strings.TrimSpace(h) == col {
			textIdx = i
			break
		}
	}

	seg, err := tokenize.NewDefault()
	must(err)

	var out []crf.Example
	for {
		rec, err := r.Read()
		if err != nil {
			break
		}
		if textIdx >= len(rec) {
			continue
		}
		text := strings.TrimSpace(rec[textIdx])
		if text == "" {
			continue
		}
		// Split on hard line breaks first (paragraph structure, when present):
		// each line is an independent document with its own final boundary.
		for _, line := range strings.Split(text, "\n") {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			toks := seg.Segment(line)
			if len(toks) == 0 {
				continue
			}
			labs := silverLabels(toks)
			out = append(out, crf.Example{Tokens: toks, Labels: labs})
		}
	}
	return out
}
