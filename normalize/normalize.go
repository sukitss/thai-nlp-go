// Package normalize will make text that "reads the same to a human" become the
// same bytes — deterministic and idempotent. Normalization is the natural first
// step of a text pipeline: consistent input makes hashing, matching and search
// consistent downstream.
//
// STATUS: planned. Normalize currently returns its input unchanged. Intended:
//   - Unicode NFC (golang.org/x/text/unicode/norm)
//   - strip zero-width: U+200B/200C/200D, BOM U+FEFF
//   - whitespace: NBSP(U+00A0)→space, collapse runs, trim
//   - canonical ordering of Thai tone marks + upper/lower vowels
//   - (optional) Thai digits ↔ Arabic, drop control chars
//
// Reference behavior: pythainlp.util.normalize.
// Contract: Normalize(Normalize(x)) == Normalize(x) (idempotent) and
// deterministic.
package normalize

// Normalize returns a canonical form of text.
//
// TODO: implement. Currently a passthrough (idempotent, but a no-op).
func Normalize(text string) string {
	return text
}
