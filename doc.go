// Package thainlp is the umbrella for thai-nlp-go — a fast, memory-conscious
// Thai NLP toolkit for Go, built for high-throughput text pipelines.
//
// The real functionality lives in sub-packages; import only what you need:
//
//	dict      shared dictionary layer (mmap flat trie, one shared instance)
//	script    split mixed-language text into runs by writing system (routing)
//	token     shared token-with-byte-offsets value used by the tokenizers
//	tokenize  word segmentation (newmm port) + char n-gram
//	cjk       Chinese word segmentation (dictionary maximal-matching, mmap)
//	jp        Japanese word segmentation (dictionary maximal-matching, mmap)
//	en        light English/Latin word tokenization (no dictionary)
//	kr        light Korean tokenization (eojeol + particle stem, no dictionary)
//	multi     one-call multilingual tokenization (detect + route th/cn/jp/kr/en)
//	          + Analyzer, one configurable pipeline for documents and queries
//	normalize Unicode/Thai text normalization (PyThaiNLP-faithful)
//	stopwords Thai/English stop-word filtering (PyThaiNLP-faithful)
//	vocab     term → sequential-id vocabulary + DF/IDF for sparse/BM25 indexing
//	sentence  whitespace sentence splitting (rule-based)
//	chunk     offset-true hierarchical chunking for RAG ingestion
//	translit  name-variant matching (MetaSound/Udom83/LK82/CompleteSoundex keys,
//	          edit distance, SoundIndex)
//	discover  propose the terms a corpus contains and the dictionary lacks
//	          (+ discover/thai for Thai, discover/auto to route by script)
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
