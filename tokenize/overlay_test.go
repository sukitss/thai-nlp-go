package tokenize

import (
	"strings"
	"testing"
)

// TestSessionOverlayRecognizesWord checks that a session-specific word is kept
// whole, while the base segmenter (which does not know it) splits it.
func TestSessionOverlayRecognizesWord(t *testing.T) {
	seg := defaultSeg(t)
	// "รักภาษา" is not a dictionary word, but "รัก" and "ภาษา" are, so the base
	// splits it in two; an overlay entry should make it one token.
	const name = "รักภาษา"

	base := seg.SegmentNoWS(name)
	if len(base) == 1 && base[0] == name {
		t.Fatalf("precondition failed: base already knows %q (%v)", name, base)
	}

	sess := seg.Session([]string{name})
	got := sess.SegmentNoWS(name)
	if len(got) != 1 || got[0] != name {
		t.Fatalf("session did not keep %q whole: %v", name, got)
	}
}

// TestSessionInContext checks the overlay word survives inside a sentence.
func TestSessionInContext(t *testing.T) {
	seg := defaultSeg(t)
	const name = "รักภาษา"
	sess := seg.Session([]string{name})
	toks := sess.SegmentNoWS("ผมชื่อรักภาษาครับ")
	if !contains(toks, name) {
		t.Fatalf("overlay word %q not found as a token in %v", name, toks)
	}
}

// TestSessionDoesNotAffectBase checks sessions are isolated from the base and
// from each other.
func TestSessionDoesNotAffectBase(t *testing.T) {
	seg := defaultSeg(t)
	before := strings.Join(seg.SegmentNoWS("รักภาษา"), "|")
	_ = seg.Session([]string{"รักภาษา"}).SegmentNoWS("รักภาษา")
	after := strings.Join(seg.SegmentNoWS("รักภาษา"), "|")
	if before != after {
		t.Fatalf("base changed after a session: %q -> %q", before, after)
	}
}

// TestOverlayEmptyIsBaseEquivalent checks an empty overlay matches the base.
func TestOverlayEmptyIsBaseEquivalent(t *testing.T) {
	seg := defaultSeg(t)
	sess := seg.Session(nil)
	for _, s := range []string{"ฉันรักภาษาไทยมาก", "abc123 ทดสอบ", "กขคง"} {
		want := strings.Join(seg.SegmentNoWS(s), "|")
		got := strings.Join(sess.SegmentNoWS(s), "|")
		if want != got {
			t.Errorf("empty overlay differs for %q: %q vs %q", s, got, want)
		}
	}
}

func contains(ss []string, x string) bool {
	for _, s := range ss {
		if s == x {
			return true
		}
	}
	return false
}
