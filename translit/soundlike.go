package translit

// Sound-alike matching for Thai names/words. It combines the phonetic keys
// (MetaSound / Udom83 / LK82) with edit distance: phonetic keys bucket
// same-sounding spellings cheaply, and edit distance catches the near-misses
// the keys drop (e.g. a differing final consonant). Use it to de-duplicate
// character-name spelling variants, match transliterations, etc.

import "unicode/utf8"

// MaxSoundLen is the maximum input length, in runes, that the fuzzy matchers
// (SoundsLike, SoundIndex.Lookup) will compare. Edit distance is O(n·m), so an
// unbounded untrusted query would let an attacker burn quadratic CPU; anything
// longer than MaxSoundLen runes is deterministically treated as not sound-alike
// (SoundsLike returns false, Lookup returns no matches). Names are short, so
// 256 runes is generous.
const MaxSoundLen = 256

// tooLongForSound reports whether s exceeds MaxSoundLen runes. Byte length is
// a free upper bound on rune count, so short strings skip the rune scan.
func tooLongForSound(s string) bool {
	return len(s) > MaxSoundLen && utf8.RuneCountInString(s) > MaxSoundLen
}

// PhoneticKeys returns the three Thai phonetic keys of s:
// [MetaSound, Udom83, LK82].
func PhoneticKeys(s string) [3]string {
	return [3]string{Key(s), Udom83(s), LK82(s)}
}

// SharesPhoneticKey reports whether a and b agree on at least one (non-empty)
// phonetic key — a strong signal that they sound alike.
func SharesPhoneticKey(a, b string) bool {
	ka, kb := PhoneticKeys(a), PhoneticKeys(b)
	for i := range ka {
		if ka[i] != "" && ka[i] == kb[i] {
			return true
		}
	}
	return false
}

// SoundsLike reports whether a and b are likely the same-sounding name: they
// share a phonetic key, OR their edit-distance Similarity is at least minSim.
// A minSim around 0.8 is a sensible default; raise it to be stricter.
// Inputs longer than MaxSoundLen runes are never sound-alike (returns false):
// this bounds the O(n·m) edit-distance cost on untrusted text.
func SoundsLike(a, b string, minSim float64) bool {
	if tooLongForSound(a) || tooLongForSound(b) {
		return false
	}
	if a == b {
		return true
	}
	return SharesPhoneticKey(a, b) || Similarity(a, b) >= minSim
}
