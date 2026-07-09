package charseg

import (
	_ "embed"
	"strings"
	"sync"
)

// charseg_model_clean.tsv is the DEFAULT, license-clean multi-domain model. It
// is trained from scratch ONLY on permissive / own / public-domain data and is
// safe to redistribute under a permissive license:
//
//   - Tatoeba Thai sentences (CC BY 2.0 FR) concatenated into synthetic
//     multi-sentence documents with exact gold boundaries  [primary signal]
//   - Wisesight social-media text (CC0)                     [realistic boundaries]
//   - thaigov-v2 government news (Public Domain)            [formal-register negatives]
//   - TLC (MIT) + Wikisource (Public Domain) literature     [formal/verse register]
//   - our own text, used with permission                    [dialogue/quote silver]
//
// It is trained from scratch: no CRF teacher, no ORCHID/TED/LST20, and no
// ShareAlike text is used. See charseg_model_clean.NOTICE for full provenance,
// licensing, the no-leakage proof, and per-domain measured quality.
//
//go:embed data/charseg_model_clean.tsv
var defaultModelData string

var (
	defaultOnce  sync.Once
	defaultModel *Model
)

// Default returns the embedded license-clean multi-domain model and is the
// recommended choice for general Thai text. Trained only on permissive / own /
// public-domain data, it is safe to redistribute under a permissive/CC0 license
// and is strongest across a fair multi-domain evaluation on macro-averaged
// boundary-F1 and space-correct accuracy — leading on dialogue / social /
// poetry / abbreviation-date cases and on a held-out hand-annotated CC0 gold
// set. The word-level CRF sub-package (sentence/crf) remains stronger on formal
// news / UD-style prose; for that register see also the opt-in Formal() model.
// For dialogue-heavy text see Dialogue().
func Default() *Model {
	defaultOnce.Do(func() {
		m, err := LoadModel(strings.NewReader(defaultModelData))
		if err != nil {
			panic("charseg: embedded default model: " + err.Error())
		}
		defaultModel = m
	})
	return defaultModel
}
