package tokenize

import (
	"strings"
	"sync"
	"testing"

	"github.com/sukitss/thai-nlp-go/dict"
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

// TestSessionWithDict checks the cached-trie path matches Session([]string).
func TestSessionWithDict(t *testing.T) {
	seg := defaultSeg(t)
	words := []string{"รักภาษา", "อาริน"}

	ov := dict.NewTrie()
	for _, w := range words {
		ov.Add(w)
	}
	cached := seg.SessionWithDict(ov)
	fromWords := seg.Session(words)

	for _, s := range []string{"ผมชื่อรักภาษาครับ", "อารินมาแล้ว", "ทดสอบ"} {
		a := strings.Join(cached.SegmentNoWS(s), "|")
		b := strings.Join(fromWords.SegmentNoWS(s), "|")
		if a != b {
			t.Fatalf("SessionWithDict != Session for %q: %q vs %q", s, a, b)
		}
	}
}

// TestSessionWithDictConcurrent shares one base dict and one cached overlay trie
// across goroutines, each with its own Segmenter — the supported caching pattern.
// Run with -race to catch data races.
func TestSessionWithDictConcurrent(t *testing.T) {
	seg := defaultSeg(t)
	ov := dict.NewTrie()
	ov.Add("รักภาษา")

	const g = 8
	var wg sync.WaitGroup
	for i := 0; i < g; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s := seg.SessionWithDict(ov) // one Segmenter per goroutine; base + ov shared
			for j := 0; j < 500; j++ {
				if !contains(s.SegmentNoWS("ผมชื่อรักภาษาครับ"), "รักภาษา") {
					t.Error("overlay word lost under concurrency")
					return
				}
			}
		}()
	}
	wg.Wait()
}

func contains(ss []string, x string) bool {
	for _, s := range ss {
		if s == x {
			return true
		}
	}
	return false
}
