package translit

// Sound-alike matching for Thai names/words. It combines the phonetic keys
// (MetaSound / Udom83 / LK82) with edit distance: phonetic keys bucket
// same-sounding spellings cheaply, and edit distance catches the near-misses
// the keys drop (e.g. a differing final consonant). Use it to de-duplicate
// character-name spelling variants, match transliterations, etc.

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
func SoundsLike(a, b string, minSim float64) bool {
	if a == b {
		return true
	}
	return SharesPhoneticKey(a, b) || Similarity(a, b) >= minSim
}
