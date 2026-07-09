// Command charsegtrain trains the character-level sentence segmenter
// (sentence/charseg) on dialogue-register silver data, saves the embedded model,
// and prints a head-to-head quality comparison (boundary-F1 + space-correct)
// against the baselines on UD_Thai-PUD and a held-out dialogue silver set.
//
// Usage:
//
//	charsegtrain -dialogue a.csv,b.csv [-eval extra.csv] [-pud th_pud.conllu] \
//	    [-bits 20] [-iters 12] [-holdout 0.15] [-out model.tsv]
package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/sukitss/thai-nlp-go/sentence"
	"github.com/sukitss/thai-nlp-go/sentence/charseg"
	"github.com/sukitss/thai-nlp-go/sentence/crf"
	"github.com/sukitss/thai-nlp-go/tokenize"
)

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "charsegtrain:", err)
		os.Exit(1)
	}
}

// --- silver labelling (orthographic; same rules as crf dialogue weak-supervision) ---

var closingSuffixes = []string{"”", "\"", "»", "」", "』", "’", ".", "!", "?", "…", "。", "！", "？"}
var openingPrefixes = []string{"“", "«", "「", "『", "„"}

func endsSentence(t string) bool {
	for _, s := range closingSuffixes {
		if strings.HasSuffix(t, s) {
			return true
		}
	}
	return false
}
func startsDialogue(t string) bool {
	for _, p := range openingPrefixes {
		if strings.HasPrefix(t, p) {
			return true
		}
	}
	return false
}

// silverSentences splits one line into silver sentence strings using the
// orthographic boundary rule over word tokens (whitespace kept).
func silverSentences(seg *tokenize.Segmenter, line string) []string {
	toks := seg.Segment(line)
	if len(toks) == 0 {
		return nil
	}
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
			labs[lastContent] = 'E'
		}
		if endsSentence(st) {
			labs[i] = 'E'
		}
		lastContent = i
	}
	if lastContent >= 0 {
		labs[lastContent] = 'E'
	}
	var out []string
	var sb strings.Builder
	for i, t := range toks {
		sb.WriteString(t)
		if labs[i] == 'E' {
			if s := strings.TrimSpace(sb.String()); s != "" {
				out = append(out, s)
			}
			sb.Reset()
		}
	}
	if s := strings.TrimSpace(sb.String()); s != "" {
		out = append(out, s)
	}
	return out
}

// readDialogue returns silver sentences grouped per source line (a "document").
func readDialogue(seg *tokenize.Segmenter, paths, col string) [][]string {
	var docs [][]string
	for _, p := range strings.Split(paths, ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		f, err := os.Open(p)
		must(err)
		r := csv.NewReader(bufio.NewReader(f))
		r.FieldsPerRecord = -1
		r.LazyQuotes = true
		header, err := r.Read()
		must(err)
		idx := 0
		for i, h := range header {
			if strings.TrimSpace(h) == col {
				idx = i
			}
		}
		nline := 0
		for {
			rec, err := r.Read()
			if err != nil {
				break
			}
			if idx >= len(rec) {
				continue
			}
			for _, line := range strings.Split(rec[idx], "\n") {
				line = strings.TrimSpace(line)
				if line == "" {
					continue
				}
				if sents := silverSentences(seg, line); len(sents) > 0 {
					docs = append(docs, sents)
					nline++
				}
			}
		}
		f.Close()
		fmt.Printf("dialogue: %d docs from %s\n", nline, p)
	}
	return docs
}

// docToExample turns a document (its silver sentences) into a char-level
// training example: concatenate sentences (single space between), boundary=true
// after the last rune of each sentence.
func docToExample(sents []string) charseg.Example {
	joined := strings.Join(sents, " ")
	runes := []rune(joined)
	bnd := make([]bool, len(runes))
	pos := 0
	for si, s := range sents {
		pos += len([]rune(s))
		if pos-1 >= 0 && pos-1 < len(bnd) {
			bnd[pos-1] = true // boundary after last rune of this sentence
		}
		if si < len(sents)-1 {
			pos++ // the joining space
		}
	}
	return charseg.Example{Runes: runes, Bnd: bnd}
}

// --- PUD ---

func readPUD(path string) []string {
	f, err := os.Open(path)
	must(err)
	defer f.Close()
	var sents []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		if strings.HasPrefix(sc.Text(), "# text = ") {
			sents = append(sents, strings.TrimSpace(strings.TrimPrefix(sc.Text(), "# text = ")))
		}
	}
	return sents
}

// --- field-based evaluation (identical method to T-112 eval.go) ---

type evalSet struct {
	doc       string
	fields    []string
	goldBreak map[int]bool
}

func buildEval(goldSents []string) evalSet {
	var fields []string
	goldBreak := map[int]bool{}
	for _, s := range goldSents {
		fs := strings.Fields(s)
		if len(fs) == 0 {
			continue
		}
		fields = append(fields, fs...)
		goldBreak[len(fields)-1] = true
	}
	delete(goldBreak, len(fields)-1)
	return evalSet{doc: strings.Join(fields, " "), fields: fields, goldBreak: goldBreak}
}

type result struct {
	name                  string
	f1, prec, rec, spaceC float64
}

func (e evalSet) eval(name string, segs []string) result {
	eng := map[int]bool{}
	idx := -1
	for _, seg := range segs {
		idx += len(strings.Fields(seg))
		eng[idx] = true
	}
	delete(eng, idx)
	tp, fp, fn, ag := 0, 0, 0, 0
	n := len(e.fields) - 1
	for g := 0; g < n; g++ {
		gb, eb := e.goldBreak[g], eng[g]
		if gb == eb {
			ag++
		}
		switch {
		case eb && gb:
			tp++
		case eb && !gb:
			fp++
		case !eb && gb:
			fn++
		}
	}
	p, r := div(tp, tp+fp), div(tp, tp+fn)
	f1 := 0.0
	if p+r > 0 {
		f1 = 2 * p * r / (p + r)
	}
	sc := 0.0
	if n > 0 {
		sc = float64(ag) / float64(n)
	}
	return result{name, f1, p, r, sc}
}

func div(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return float64(a) / float64(b)
}

func trainOn(docs [][]string, bits, iters int) (*charseg.Model, int, int) {
	examples := make([]charseg.Example, 0, len(docs))
	var totRune, totBnd int
	for _, d := range docs {
		ex := docToExample(d)
		totRune += len(ex.Runes)
		for _, b := range ex.Bnd {
			if b {
				totBnd++
			}
		}
		examples = append(examples, ex)
	}
	return charseg.Train(examples, charseg.TrainConfig{Bits: bits, Iters: iters}), totRune, totBnd
}

func flatten(docs [][]string) []string {
	var out []string
	for _, d := range docs {
		out = append(out, d...)
	}
	return out
}

// docSplit partitions docs into disjoint train/held-out by a document stride
// (whole documents/chapters kept together — no within-document leakage).
func docSplit(docs [][]string, holdout float64) (train, held [][]string) {
	stride := int(1.0 / holdout)
	if stride < 2 {
		stride = 2
	}
	for i, d := range docs {
		if i%stride == 0 {
			held = append(held, d)
		} else {
			train = append(train, d)
		}
	}
	return
}

func main() {
	dialogueCSV := flag.String("dialogue", "", "dialogue-register CSV(s) — TRAIN (required)")
	evalCSV := flag.String("eval", "", "separate-FILE held-out dialogue CSV(s)")
	pud := flag.String("pud", "", "UD_Thai-PUD conllu (TEST ONLY, never trained on)")
	col := flag.String("col", "text", "text column")
	bits := flag.Int("bits", 20, "log2 weight-table size")
	iters := flag.Int("iters", 12, "perceptron iterations")
	holdout := flag.Float64("holdout", 0.15, "internal doc-disjoint held-out fraction")
	out := flag.String("out", "sentence/charseg/data/charseg_model_dialogue.tsv", "model output path")
	sweep := flag.Bool("sweep", false, "sweep weight-table size (bits) for quality/speed")
	flag.Parse()

	if *dialogueCSV == "" {
		fmt.Fprintln(os.Stderr, "need -dialogue <csv[,csv...]>")
		os.Exit(2)
	}

	seg, err := tokenize.NewDefault()
	must(err)

	trainDocs := readDialogue(seg, *dialogueCSV, *col)
	if len(trainDocs) == 0 {
		fmt.Fprintln(os.Stderr, "no dialogue docs")
		os.Exit(1)
	}
	var sepHeld [][]string
	if *evalCSV != "" {
		sepHeld = readDialogue(seg, *evalCSV, *col) // fully separate file (disjoint)
	}

	// ---- shipped model: trained on ALL of -dialogue; held-out numbers below use
	// a fully separate file and PUD (never trained on). ----
	model, totRune, totBnd := trainOn(trainDocs, *bits, *iters)
	fmt.Printf("TRAIN (shipped): %d docs, %d runes, %d boundaries (%.2f%% of runes) from %s\n",
		len(trainDocs), totRune, totBnd, 100*float64(totBnd)/float64(totRune), *dialogueCSV)

	if *out != "" {
		f, err := os.Create(*out)
		must(err)
		must(model.Save(f))
		f.Close()
		fi, _ := os.Stat(*out)
		fmt.Printf("saved model -> %s (%d KB)\n", *out, fi.Size()/1024)
	}

	report := func(title string, es evalSet, charSplit func(string) []string) {
		engines := []struct {
			name  string
			split func(string) []string
		}{
			{"whitespace", sentence.Split},
			{"heuristic", sentence.SplitHeuristic},
			{"crf.Default", crf.Split},
			{"crf.Dialogue", func(s string) []string { return crf.Dialogue().Split(s) }},
			{"charseg", charSplit},
		}
		fmt.Printf("\n=== %s (%d fields, %d gold boundaries) ===\n", title, len(es.fields), len(es.goldBreak))
		fmt.Printf("%-14s %8s %8s %8s %8s\n", "engine", "F1", "P", "R", "space-ok")
		for _, e := range engines {
			r := es.eval(e.name, e.split(es.doc))
			fmt.Printf("%-14s %8.4f %8.4f %8.4f %8.4f\n", r.name, r.f1, r.prec, r.rec, r.spaceC)
		}
	}

	if *pud != "" {
		report("UD_Thai-PUD [TEST, out-of-domain]", buildEval(readPUD(*pud)), model.Split)
	}
	if len(sepHeld) > 0 {
		report("Dialogue SEPARATE-FILE held-out (matches shipped model)",
			buildEval(flatten(sepHeld)), model.Split)
	}

	// ---- internal doc-disjoint held-out: separate model trained ONLY on the
	// train split so the reported number never touched training. ----
	inTrain, inHeld := docSplit(trainDocs, *holdout)
	inModel, _, _ := trainOn(inTrain, *bits, *iters)
	fmt.Printf("\n(internal split: train=%d docs, held-out=%d docs, doc-disjoint)", len(inTrain), len(inHeld))
	report("Dialogue INTERNAL held-out (doc-disjoint, no-leak model)",
		buildEval(flatten(inHeld)), inModel.Split)

	if *sweep && len(sepHeld) > 0 && *pud != "" {
		fmt.Printf("\n=== bits sweep (table size = 4*2^bits bytes; quality on separate-file + PUD) ===\n")
		fmt.Printf("%-6s %10s %10s %10s %12s\n", "bits", "F1_dialog", "F1_PUD", "model_KB", "us/Split")
		sepEval := buildEval(flatten(sepHeld))
		pudEval := buildEval(readPUD(*pud))
		docs := sepEval.doc
		for _, bt := range []int{12, 14, 16, 18, 20, 22} {
			m, _, _ := trainOn(trainDocs, bt, *iters)
			fn := m.Split(docs) // warm
			_ = fn
			iterN := 2000
			t0 := time.Now()
			for i := 0; i < iterN; i++ {
				m.Split(docs)
			}
			us := float64(time.Since(t0).Microseconds()) / float64(iterN)
			var buf strings.Builder
			m.Save(&buf)
			kb := len(buf.String()) / 1024
			f1n := sepEval.eval("c", m.Split(sepEval.doc)).f1
			f1p := pudEval.eval("c", m.Split(pudEval.doc)).f1
			fmt.Printf("%-6d %10.4f %10.4f %10d %12.1f\n", bt, f1n, f1p, kb, us)
		}
	}
}
