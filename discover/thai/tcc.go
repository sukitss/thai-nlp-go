package thai

import (
	"sync"

	"github.com/sukitss/thai-nlp-go/tokenize"
)

// The TCC matcher is stateless and read-only; one is enough for the process.
var (
	tccOnce sync.Once
	tcc     *tokenize.TCC
)

func sharedTCC() *tokenize.TCC {
	tccOnce.Do(func() { tcc = tokenize.NewTCC() })
	return tcc
}
