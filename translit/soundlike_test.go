package translit

import "testing"

func TestSoundsLike(t *testing.T) {
	// same-sounding variants -> true
	likePairs := [][2]string{
		{"ทองดี", "ทองดา"},   // share a phonetic key
		{"รัก", "ลัก"},       // ร/ล fold in the keys
		{"อุจิวะ", "อุจิฮะ"}, // keys differ but 1 edit -> similarity ~0.83
		{"กิตติ", "กิติ"},    // 1 edit
	}
	for _, p := range likePairs {
		if !SoundsLike(p[0], p[1], 0.8) {
			t.Errorf("SoundsLike(%q,%q) = false, want true (keys=%v/%v sim=%.2f)",
				p[0], p[1], PhoneticKeys(p[0]), PhoneticKeys(p[1]), Similarity(p[0], p[1]))
		}
	}
	// clearly different -> false
	diffPairs := [][2]string{
		{"แมว", "รถยนต์"},
		{"ประเทศไทย", "กิน"},
	}
	for _, p := range diffPairs {
		if SoundsLike(p[0], p[1], 0.8) {
			t.Errorf("SoundsLike(%q,%q) = true, want false", p[0], p[1])
		}
	}
}

func TestSharesPhoneticKey(t *testing.T) {
	if !SharesPhoneticKey("ทองดี", "ทองดา") {
		t.Error("ทองดี/ทองดา should share a phonetic key")
	}
	if SharesPhoneticKey("แมว", "ปลา") {
		t.Error("แมว/ปลา should not share a phonetic key")
	}
}
