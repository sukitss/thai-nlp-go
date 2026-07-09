package charseg

import (
	_ "embed"
	"strings"
	"sync"
)

// charseg_model_formal.tsv is the FORMAL-register model, produced by knowledge
// distillation from the SaT / wtpsplit "sat-12l-sm" neural teacher (MIT-licensed
// code and weights, Minixhofer et al.) applied to permissive Thai text
// (thaigov-v2 Public Domain + Wisesight CC0). Distillation transfers the
// teacher's segmentation *function*, not its data, so the student weights are
// safe to redistribute under a permissive license — the char-level engine,
// trainer and candidate definition remain our own original code.
//
// It is a specialist. Evaluated at its FormalTau operating point it edges the
// permissive Default() on leave-one-register-out honest macro boundary-F1
// (0.756 vs 0.732) and is stronger on formal / academic / news / abbreviation
// register, but it over-segments informal dialogue and chat — for dialogue-heavy
// text prefer Default() (recommended) or Dialogue(). See
// charseg_model_formal.NOTICE for full provenance, the no-leakage proof, and
// per-register measured quality.
//
//go:embed data/charseg_model_formal.tsv
var formalModelData string

var (
	formalOnce  sync.Once
	formalModel *Model
)

// Formal returns the embedded formal-register model, distilled from the SaT
// (wtpsplit) neural teacher on permissive Thai text. It is an opt-in specialist
// for formal / academic / news prose; apply its operating point with FormalSplit
// (or Formal().SplitTau(text, FormalTau)). For general or dialogue-heavy text
// use Default() (recommended) or Dialogue().
func Formal() *Model {
	formalOnce.Do(func() {
		m, err := LoadModel(strings.NewReader(formalModelData))
		if err != nil {
			panic("charseg: embedded formal model: " + err.Error())
		}
		formalModel = m
	})
	return formalModel
}

// FormalTau is the operating point (τ) that maximizes macro boundary-F1 for the
// Formal model. It is strongly recall-leaning and was selected by leave-one-
// register-out cross-validation — the same τ was chosen on every fold, and the
// honest (LODO) and in-sample macro scores coincide, so it generalizes rather
// than overfits the evaluation set. Retraining the model requires re-tuning it.
const FormalTau = -26.0

// FormalSplit segments text with the Formal model at its FormalTau operating
// point. Equivalent to Formal().SplitTau(text, FormalTau).
func FormalSplit(text string) []string { return Formal().SplitTau(text, FormalTau) }
