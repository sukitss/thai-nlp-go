// Package sentence splits Thai text into segments. Thai has no sentence-final
// punctuation, so these are rule-based splitters on whitespace (Thai commonly
// uses spaces at phrase/sentence boundaries). They are deterministic and fast.
//
// They match PyThaiNLP's non-ML engines: Split == sent_tokenize(engine=
// "whitespace+newline"), SplitSpaces == engine="whitespace". For higher-quality
// ML segmentation (PyThaiNLP's default crfcut) compose a model separately.
package sentence

import (
	"regexp"
	"strings"
)

var reSpaces = regexp.MustCompile(" +")

// Split splits text on any run of whitespace (spaces, tabs, newlines) and drops
// empty segments — equivalent to sent_tokenize(engine="whitespace+newline").
func Split(text string) []string {
	return strings.Fields(text)
}

// SplitSpaces splits text on runs of ASCII space only, preserving empty leading,
// trailing and adjacent segments — equivalent to engine="whitespace". Empty
// input returns nil.
func SplitSpaces(text string) []string {
	if text == "" {
		return nil
	}
	return reSpaces.Split(text, -1)
}
