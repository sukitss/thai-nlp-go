// Package translit matches Thai names/words that sound alike but are spelled
// differently, by reducing them to a phonetic key. Words with the same key are
// likely variants of the same name — useful for de-duplicating spelling
// variants, name search and entity matching.
//
// Key uses MetaSound, a Thai phonetic algorithm (a faithful port of PyThaiNLP's
// metasound). It keeps the first consonant and encodes the rest by sound, so
// e.g. "ทองดี" and "ทองดา", or "บ้าน" and "บาน", share a key. It is a heuristic
// baseline: distinct final consonants (e.g. "อุจิวะ" vs "อุจิฮะ") still differ —
// combine with fuzzy matching / an alias dictionary for those.
package translit

import "strings"

const (
	consThanthakhat = "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ์"
	thanthakhat     = '์' // U+0E4C, silences the preceding consonant

	sndK  = "กขฃคฆฅ"         // -> 1
	sndD  = "จฉชฌซฐทฒดฎตสศษ" // -> 2
	sndB  = "ฟฝพผภบป"        // -> 3
	sndNG = "ง"              // -> 4
	sndN  = "ลฬรนณฦญ"        // -> 5
	sndM  = "ม"              // -> 6
	sndY  = "ย"              // -> 7
	sndW  = "ว"              // -> 8
)

// Key returns the default phonetic key (MetaSound, length 4) for name.
func Key(name string) string { return Metasound(name, 4) }

// Metasound returns the MetaSound phonetic code of text with the given length
// (the code keeps the first consonant and encodes up to length-1 more by sound,
// padding with '0'). Faithful port of PyThaiNLP metasound.
func Metasound(text string, length int) string {
	if text == "" || length <= 0 {
		return ""
	}
	// keep only consonants and thanthakhat
	chars := make([]rune, 0, len(text))
	for _, ch := range text {
		if strings.ContainsRune(consThanthakhat, ch) {
			chars = append(chars, ch)
		}
	}
	// thanthakhat silences itself and the preceding consonant
	for i := 0; i < len(chars); i++ {
		if chars[i] == thanthakhat {
			if i > 0 {
				chars[i-1] = ' '
			}
			chars[i] = ' '
		}
	}
	if len(chars) > length {
		chars = chars[:length]
	}
	// encode everything after the first character by sound
	for i := 1; i < len(chars); i++ {
		chars[i] = encode(chars[i])
	}
	for len(chars) < length {
		chars = append(chars, '0')
	}
	return string(chars)
}

func encode(r rune) rune {
	switch {
	case strings.ContainsRune(sndK, r):
		return '1'
	case strings.ContainsRune(sndD, r):
		return '2'
	case strings.ContainsRune(sndB, r):
		return '3'
	case strings.ContainsRune(sndNG, r):
		return '4'
	case strings.ContainsRune(sndN, r):
		return '5'
	case strings.ContainsRune(sndM, r):
		return '6'
	case strings.ContainsRune(sndY, r):
		return '7'
	case strings.ContainsRune(sndW, r):
		return '8'
	default:
		return '0'
	}
}
