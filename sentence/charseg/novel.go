package charseg

import (
	_ "embed"
	"strings"
	"sync"
)

// charseg_model.tsv is the NOVEL-domain model: trained by weak supervision on
// our own Thai web-novel text (dialogue/quote silver). It is a specialist —
// strongest on novel dialogue and quotes, but it transfers poorly to formal
// prose. It is trained on data we own (no third-party corpus, no NC/ND/SA), so
// it is safe to redistribute. See charseg_model.NOTICE for provenance and
// measured quality. For general multi-domain text prefer Default().
//
//go:embed data/charseg_model.tsv
var novelModelData string

var (
	novelOnce  sync.Once
	novelModel *Model
)

// Novel returns the embedded NOVEL-domain model (own web-novel silver data). It
// is an opt-in specialist for novel/dialogue-heavy text; for general
// multi-domain text use Default() (the recommended permissive model).
func Novel() *Model {
	novelOnce.Do(func() {
		m, err := LoadModel(strings.NewReader(novelModelData))
		if err != nil {
			panic("charseg: embedded novel model: " + err.Error())
		}
		novelModel = m
	})
	return novelModel
}
