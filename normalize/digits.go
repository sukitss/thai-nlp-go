package normalize

import "unicode/utf8"

// Thai digit block ๐..๙ (U+0E50–U+0E59).
const (
	thaiDigitZero = 0x0E50
	thaiDigitNine = 0x0E59
)

// DigitsToArabic maps Thai digits ๐-๙ (U+0E50–U+0E59) to ASCII 0-9 and leaves
// every other byte untouched (including invalid UTF-8, which is copied
// verbatim). It is deliberately separate from Normalize, which stays faithful
// to PyThaiNLP and does not fold digits — apply it after Normalize when an
// index should treat "๕" and "5" as the same term.
//
// Zero-alloc when the input contains no Thai digit: the input string is
// returned as-is.
func DigitsToArabic(s string) string {
	// Fast scan for the first Thai digit. ASCII bytes are skipped without a
	// rune decode; a Thai digit's UTF-8 form is E0 B9 90..99, so anything
	// below 0xE0 can't start one either.
	i := 0
	for i < len(s) {
		c := s[i]
		if c < 0xE0 {
			i++
			continue
		}
		r, sz := utf8.DecodeRuneInString(s[i:])
		if r >= thaiDigitZero && r <= thaiDigitNine {
			break
		}
		i += sz
	}
	if i == len(s) {
		return s // no Thai digit: zero-alloc
	}
	// A Thai digit is 3 bytes, its ASCII replacement 1, so len(s) always fits.
	b := make([]byte, 0, len(s))
	b = append(b, s[:i]...)
	for i < len(s) {
		if c := s[i]; c < 0xE0 {
			b = append(b, c)
			i++
			continue
		}
		r, sz := utf8.DecodeRuneInString(s[i:])
		if r >= thaiDigitZero && r <= thaiDigitNine {
			b = append(b, byte('0'+r-thaiDigitZero))
		} else {
			b = append(b, s[i:i+sz]...) // raw bytes: invalid UTF-8 passes through
		}
		i += sz
	}
	return string(b)
}
