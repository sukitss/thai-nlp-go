// Package sentence will split Thai text into sentences. Thai has no clear
// sentence-final punctuation, so this needs whitespace + context (or a model).
// Useful for sentence-aware chunking and alignment.
//
// STATUS: planned. Options: rule-based on spaces + cues, ICU, or a CRF like
// pythainlp sent_tokenize. Keep it deterministic; if a model is needed, load it
// once and share it (see package dict).
package sentence

// Split returns the sentences of text.
//
// TODO: implement. This stub returns the whole input as one sentence.
func Split(text string) []string {
	if text == "" {
		return nil
	}
	return []string{text}
}
