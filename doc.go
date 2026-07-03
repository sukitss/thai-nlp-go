// Package thainlp is the umbrella for thai-nlp-go — a fast, memory-conscious
// Thai NLP toolkit for Go, built for high-throughput text pipelines.
//
// The real functionality lives in sub-packages; import only what you need:
//
//	dict      shared dictionary layer (mmap flat trie, one shared instance)
//	script    split mixed-language text into runs by writing system (routing)
//	tokenize  word segmentation (newmm port) + char n-gram
//	normalize Unicode/Thai text normalization (PyThaiNLP-faithful)
//	stopwords Thai/English stop-word filtering (PyThaiNLP-faithful)
//	sentence  whitespace sentence splitting (rule-based)
//	translit  name-variant matching (MetaSound/Udom83/LK82/CompleteSoundex keys,
//	          edit distance, SoundIndex)
//
// Design rule #1: performance and memory. Components share a single
// memory-mapped dictionary via dict.Default rather than each loading its own —
// see ARCHITECTURE.md.
//
// # Quick start
//
//	seg, _ := tokenize.NewDefault()
//	seg.SegmentNoWS("ฉันรักภาษาไทยมาก")        // ["ฉัน" "รัก" "ภาษาไทย" "มาก"]
//	seg.SegmentBytes("ฉันรักภาษาไทยมาก", ' ')  // []byte, ready for a keyword index
//
// # Credits & license
//
// The tokenization algorithm and dictionary are from PyThaiNLP
// (https://github.com/PyThaiNLP/pythainlp; Apache-2.0 / CC0-1.0). This module is
// released under Apache-2.0. See LICENSE and NOTICE.
package thainlp
