// Command thainlp is the CLI for thai-nlp-go.
//
// By default it tokenizes Thai text from stdin (one line in, one line of
// space-joined tokens out). Subcommands cover dictionary maintenance and,
// over time, the other NLP tools in this module.
//
// Usage:
//
//	thainlp [flags] < input.txt > tokens.txt        # tokenize (default)
//	thainlp build -dict data/words_th.txt -out x.fdt # (re)build a flat trie
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/sukitss/thai-nlp-go/dict"
	"github.com/sukitss/thai-nlp-go/tokenize"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "build" {
		buildCmd(os.Args[2:])
		return
	}

	dictPath := flag.String("dict", "", "flat-trie file (.fdt) to load; empty = use embedded dictionary")
	mmapFlag := flag.Bool("mmap", false, "with -dict, mmap the file instead of reading it into memory")
	keepWS := flag.Bool("keep-ws", false, "keep whitespace tokens (default drops them)")
	overlay := flag.String("overlay", "", "comma-separated extra words for this run (session overlay)")
	timing := flag.Bool("timing", false, "print load time to stderr")
	flag.Parse()

	t0 := time.Now()
	seg, err := load(*dictPath, *mmapFlag)
	must(err)
	if *overlay != "" {
		seg = seg.Session(strings.Split(*overlay, ","))
	}
	if *timing {
		fmt.Fprintf(os.Stderr, "[load] %v\n", time.Since(t0))
	}

	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024)
	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()

	for in.Scan() {
		line := in.Text()
		if *keepWS {
			out.WriteString(strings.Join(seg.Segment(line), " "))
		} else {
			out.Write(seg.SegmentBytes(line, ' '))
		}
		out.WriteByte('\n')
	}
	must(in.Err())
}

func load(dictPath string, useMmap bool) (*tokenize.Segmenter, error) {
	if dictPath == "" {
		return tokenize.NewDefault() // shared embedded dictionary
	}
	var (
		ft  *dict.FlatTrie
		err error
	)
	if useMmap {
		ft, err = dict.OpenFlat(dictPath)
	} else {
		ft, err = dict.ReadFlat(dictPath)
	}
	if err != nil {
		return nil, err
	}
	return tokenize.New(ft), nil
}

func buildCmd(args []string) {
	fs := flag.NewFlagSet("build", flag.ExitOnError)
	dictPath := fs.String("dict", "dict/data/words_th.txt", "dictionary text file (one word per line)")
	outPath := fs.String("out", "dict/data/words_th.fdt", "output flat-trie file")
	fs.Parse(args)

	t0 := time.Now()
	t, err := dict.LoadDict(*dictPath)
	must(err)
	must(dict.BuildFlatFromTrie(t, *outPath))
	fi, _ := os.Stat(*outPath)
	fmt.Fprintf(os.Stderr, "[build] %d words → %s (%d bytes) in %v\n",
		t.Len(), *outPath, fi.Size(), time.Since(t0))
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "thainlp:", err)
		os.Exit(1)
	}
}
