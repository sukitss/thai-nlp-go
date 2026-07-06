// Command crftrain trains and evaluates a CRF Thai sentence segmenter on
// CoNLL-U corpora (e.g. Universal Dependencies Thai), and prints a before/after
// comparison against the baselines (whitespace split + the embedded crfcut).
//
// Usage:
//
//	crftrain -train th_tud-ud-train.conllu -eval th_tud-ud-test.conllu[,more] [-iters 10] [-out model.txt]
//
// Documents are reconstructed from CoNLL-U with SpaceAfter=No honored (so real
// spacing — including sentences that end WITHOUT a space — is preserved), and
// sentences are grouped into documents by the "# filename" field.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/sukitss/thai-nlp-go/sentence/crf"
)

func main() {
	trainPath := flag.String("train", "", "CoNLL-U training file")
	evalPaths := flag.String("eval", "", "comma-separated CoNLL-U eval files")
	iters := flag.Int("iters", 10, "perceptron iterations")
	out := flag.String("out", "", "write trained model to this file")
	novelCSV := flag.String("novel", "", "comma-separated novel CSV files (weak-supervision silver corpus)")
	novelEval := flag.String("noveleval", "", "optional separate novel CSV(s) for held-out eval (else internal split)")
	novelCol := flag.String("col", "text", "text column name in the novel CSV")
	holdout := flag.Float64("holdout", 0.2, "held-out fraction for -novel mode (ignored if -noveleval set)")
	flag.Parse()

	if *novelCSV != "" {
		runNovel(*novelCSV, *novelEval, *novelCol, *holdout, *iters, *out)
		return
	}

	if *trainPath == "" {
		fmt.Fprintln(os.Stderr, "need -train (or -novel)")
		os.Exit(2)
	}
	train := readDocs(*trainPath)
	fmt.Printf("train: %d docs (%s)\n", len(train), *trainPath)

	model := crf.Train(train, *iters)

	if *out != "" {
		f, err := os.Create(*out)
		must(err)
		must(model.Save(f))
		f.Close()
		fmt.Printf("saved model -> %s\n", *out)
	}

	for _, ep := range strings.Split(*evalPaths, ",") {
		ep = strings.TrimSpace(ep)
		if ep == "" {
			continue
		}
		gold := readDocs(ep)
		fmt.Printf("\n=== eval: %s (%d docs) ===\n", ep, len(gold))
		fmt.Printf("  whitespace  : %s\n", crf.Eval(crf.WhitespaceBaseline, gold))
		fmt.Printf("  crfcut(TED) : %s\n", crf.Eval(crf.Default().Labels, gold))
		fmt.Printf("  trained     : %s\n", crf.Eval(model.Labels, gold))
	}
}

// readDocs parses a CoNLL-U file into documents (grouped by "# filename"),
// reconstructing token streams with space tokens per SpaceAfter and labelling
// the last content token of each sentence 'E'.
func readDocs(path string) []crf.Example {
	f, err := os.Open(path)
	must(err)
	defer f.Close()

	type tok struct {
		form     string
		spaceAft bool
	}
	type sent struct {
		file string
		toks []tok
	}
	var sents []sent
	var cur []tok
	file := ""
	flush := func() {
		if len(cur) > 0 {
			sents = append(sents, sent{file, cur})
			cur = nil
		}
	}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<16), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if line == "" {
			flush()
			continue
		}
		if strings.HasPrefix(line, "#") {
			if i := strings.Index(line, "# filename ="); i == 0 {
				file = strings.TrimSpace(line[len("# filename ="):])
			}
			continue
		}
		c := strings.Split(line, "\t")
		if len(c) < 10 {
			continue
		}
		id := c[0]
		if strings.ContainsAny(id, "-.") { // multiword-token range / empty node
			continue
		}
		spaceAfter := !strings.Contains(c[9], "SpaceAfter=No")
		cur = append(cur, tok{c[1], spaceAfter})
	}
	flush()

	// group consecutive sentences with the same filename into documents
	var docs []crf.Example
	var toks []string
	var labs []byte
	curFile := ""
	nSent := 0
	const maxSentsPerDoc = 12 // keep docs reasonable (and split filename-less corpora)
	emit := func() {
		if len(toks) > 0 {
			docs = append(docs, crf.Example{Tokens: toks, Labels: labs})
			toks, labs = nil, nil
		}
		nSent = 0
	}
	for _, s := range sents {
		if s.file != curFile || nSent >= maxSentsPerDoc {
			emit()
			curFile = s.file
		}
		nSent++
		lastContent := -1
		for _, t := range s.toks {
			toks = append(toks, t.form)
			labs = append(labs, 'I')
			lastContent = len(toks) - 1
			if t.spaceAft {
				toks = append(toks, " ")
				labs = append(labs, 'I')
			}
		}
		if lastContent >= 0 {
			labs[lastContent] = 'E' // sentence end
		}
	}
	emit()
	return docs
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "crftrain:", err)
		os.Exit(1)
	}
}
