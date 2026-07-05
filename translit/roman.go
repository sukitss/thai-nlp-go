package translit

import "strings"

// Cross-lingual name matching: a translated Chinese novel spells a character's
// name three ways — the Hanzi (聂力), the pinyin ("Nie Li"), and a Thai
// transliteration (เนี่ยหลี่). PinyinToThai turns a romanized (pinyin) name into
// an approximate Thai spelling so it lands in the same SoundIndex phonetic
// bucket as the hand-written Thai form. It is a phonetic bridge for matching,
// not a faithful transliteration — precision beyond common syllables needs a
// corpus (see the package tests for the coverage that is verified).
//
// Japanese romaji is a separate table (a follow-up); this covers Mandarin
// pinyin, the dominant case for translated-novel character names.

// pinyin initials → Thai consonant.
var pinyinInitials = map[string]string{
	"zh": "จ", "ch": "ช", "sh": "ช",
	"b": "บ", "p": "พ", "m": "ม", "f": "ฟ",
	"d": "ด", "t": "ท", "n": "น", "l": "ล",
	"g": "ก", "k": "ค", "h": "ห",
	"j": "จ", "q": "ช", "x": "ซ",
	"r": "ร", "z": "จ", "c": "ช", "s": "ส",
	"y": "ย", "w": "ว",
}

// pinyin finals → Thai (pre goes BEFORE the consonant — Thai leading vowels
// เ/แ/โ/ไ wrap the consonant — post goes after).
type finalTh struct{ pre, post string }

var pinyinFinals = map[string]finalTh{
	"a": {"", "า"}, "o": {"", "อ"}, "e": {"เ", "อ"}, "ê": {"เ", "ะ"},
	"ai": {"", "าย"}, "ei": {"เ", "ย"}, "ao": {"", "าว"}, "ou": {"โ", "ว"},
	"an": {"", "าน"}, "en": {"เ", "ิน"}, "ang": {"", "าง"}, "eng": {"เ", "ิง"},
	"ong": {"", "ง"}, "er": {"เ", "อ"},
	"i": {"", "ี"}, "ia": {"เ", "ีย"}, "ie": {"เ", "ีย"}, "iao": {"เ", "ียว"},
	"iu": {"", "ีว"}, "ian": {"เ", "ียน"}, "in": {"", "ิน"}, "iang": {"เ", "ียง"},
	"ing": {"", "ิง"}, "iong": {"", "ียง"},
	"u": {"", "ู"}, "ua": {"", "วา"}, "uo": {"", "ัว"}, "uai": {"", "วาย"},
	"ui": {"", "ุย"}, "uan": {"", "วน"}, "un": {"", "ุน"}, "uang": {"", "วาง"},
	"ueng": {"เ", "ิง"},
	"ü":    {"", "วี"}, "üe": {"เ", "วีย"}, "üan": {"เ", "วียน"}, "ün": {"", "วิน"},
}

// splitPinyinSyllable splits a lower-cased pinyin syllable (no tone marks) into
// its initial and final. Returns ok=false if the final is unknown.
func splitPinyinSyllable(syl string) (thai string, ok bool) {
	// try two-letter initials first (zh/ch/sh), then one-letter.
	for _, ln := range []int{2, 1} {
		if len(syl) >= ln {
			ini := syl[:ln]
			if c, has := pinyinInitials[ini]; has {
				if f, hasF := pinyinFinals[syl[ln:]]; hasF {
					return f.pre + c + f.post, true
				}
			}
		}
	}
	// no initial (syllable is a bare final: a, an, ai, e, …)
	if f, hasF := pinyinFinals[syl]; hasF {
		return f.pre + f.post, true
	}
	return "", false
}

// PinyinToThai converts a pinyin name (one or more syllables separated by
// spaces/hyphens, tone marks ignored) to an approximate Thai spelling. Unknown
// syllables are dropped. Returns "" if nothing converts.
func PinyinToThai(name string) string {
	name = strings.ToLower(strings.TrimSpace(name))
	if name == "" {
		return ""
	}
	fields := strings.FieldsFunc(name, func(r rune) bool { return r == ' ' || r == '-' || r == '\'' })
	var b strings.Builder
	for _, f := range fields {
		if th, ok := splitPinyinSyllable(stripTones(f)); ok {
			b.WriteString(th)
		}
	}
	return b.String()
}

// stripTones drops pinyin tone diacritics, folding to the base vowel.
func stripTones(s string) string {
	var b strings.Builder
	for _, r := range s {
		if base, ok := toneBase[r]; ok {
			b.WriteRune(base)
		} else {
			b.WriteRune(r)
		}
	}
	return b.String()
}

var toneBase = map[rune]rune{
	'ā': 'a', 'á': 'a', 'ǎ': 'a', 'à': 'a',
	'ē': 'e', 'é': 'e', 'ě': 'e', 'è': 'e',
	'ī': 'i', 'í': 'i', 'ǐ': 'i', 'ì': 'i',
	'ō': 'o', 'ó': 'o', 'ǒ': 'o', 'ò': 'o',
	'ū': 'u', 'ú': 'u', 'ǔ': 'u', 'ù': 'u',
	'ǖ': 'ü', 'ǘ': 'ü', 'ǚ': 'ü', 'ǜ': 'ü',
}

// isASCIILetters reports whether s contains ASCII letters (a romanized form
// worth bridging to Thai). Used to decide when to add cross-lingual keys.
func isASCIILetters(s string) bool {
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
			return true
		}
	}
	return false
}

// crossSim is Similarity that also bridges a romanized (pinyin) query to Thai:
// it returns the max of the raw similarity and the similarity of the query's
// Thai transliteration, so "Nie Li" scores high against "เนี่ยหลี่".
func crossSim(query, name string) float64 {
	s := Similarity(query, name)
	if isASCIILetters(query) {
		if th := PinyinToThai(query); th != "" {
			if s2 := Similarity(th, name); s2 > s {
				s = s2
			}
		}
	}
	return s
}
