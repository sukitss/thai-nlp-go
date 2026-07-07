package query

import (
	"bufio"
	_ "embed"
	"strings"
	"sync"

	"github.com/sukitss/thai-nlp-go/stopwords"
)

//go:embed data/query_filler_th.txt
var fillerData string

var (
	fillerOnce   sync.Once
	fillerShared *stopwords.Set
)

// fillerSet returns the shared conversational query-filler stop set (politeness
// and question particles, generic verbs of intent). Loaded once; read-only.
func fillerSet() *stopwords.Set {
	fillerOnce.Do(func() {
		var words []string
		sc := bufio.NewScanner(strings.NewReader(fillerData))
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			words = append(words, strings.Fields(line)...)
		}
		fillerShared = stopwords.New(words...)
	})
	return fillerShared
}
