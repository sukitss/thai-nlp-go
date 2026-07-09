package charseg

import (
	_ "embed"
	"strings"
	"sync"
)

// charseg_model_dialogue.tsv is a specialist model for dialogue- and quote-heavy
// prose (conversational text, quoted speech). It is trained by weak supervision
// on text we own (used with permission), from orthographic silver signals
// (closing quote / terminal punctuation / a token before an opening quote). It
// is strongest on dialogue and quoted speech but transfers poorly to formal
// prose. It is trained only on our own text (no third-party corpus, no NC/ND/SA),
// so it is safe to redistribute. See charseg_model_dialogue.NOTICE for provenance
// and measured quality. For general multi-domain text prefer Default().
//
//go:embed data/charseg_model_dialogue.tsv
var dialogueModelData string

var (
	dialogueOnce  sync.Once
	dialogueModel *Model
)

// Dialogue returns the embedded dialogue-register model. It is an opt-in
// specialist for dialogue- and quote-heavy prose; for general multi-domain text
// use Default() (the recommended permissive model).
func Dialogue() *Model {
	dialogueOnce.Do(func() {
		m, err := LoadModel(strings.NewReader(dialogueModelData))
		if err != nil {
			panic("charseg: embedded dialogue model: " + err.Error())
		}
		dialogueModel = m
	})
	return dialogueModel
}
