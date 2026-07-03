package tokenize

import (
	"slices"
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

var _ dict.Weighter = (*OverlayDict)(nil)

// prefixerOnly hides PrefixWeights, modelling a base dictionary that only
// implements dict.Prefixer.
type prefixerOnly struct{ t *dict.Trie }

func (p prefixerOnly) PrefixLens(text []rune, start int, out []int) []int {
	return p.t.PrefixLens(text, start, out)
}

// weightedFlat builds a FlatTrie base via the in-memory serialization path.
func weightedFlat(t *testing.T, tr *dict.Trie) *dict.FlatTrie {
	t.Helper()
	data, err := dict.FlatBytes(tr)
	if err != nil {
		t.Fatal(err)
	}
	ft, err := dict.FromBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	return ft
}

func checkPrefixWeights(t *testing.T, name string, o *OverlayDict, text []rune, start int, wantLen, wantW []int32) {
	t.Helper()
	gotL, gotW := o.PrefixWeights(text, start, nil, nil)
	if !slices.Equal(gotL, wantLen) || !slices.Equal(gotW, wantW) {
		t.Errorf("%s: PrefixWeights at %d = (%v,%v), want (%v,%v)", name, start, gotL, gotW, wantLen, wantW)
	}
}

// TestOverlayPrefixWeights pins the merge/shadowing semantics against a
// weighted FlatTrie base: overlay-only words carry the overlay weight,
// base-only words pass the base weight through, and a word in both takes the
// overlay weight only when the overlay is weighted (an unweighted overlay has
// no weight opinion).
func TestOverlayPrefixWeights(t *testing.T) {
	baseTr := dict.NewTrie()
	baseTr.AddWeighted("กา", 10)
	baseTr.AddWeighted("กาแฟ", 20)
	baseTr.AddWeighted("แฟน", 5)
	base := weightedFlat(t, baseTr)

	overlayWords := []string{"กาแฟดำ", "กา", "อาริน"}
	weighted := dict.NewTrie()
	weighted.AddWeighted("กาแฟดำ", 7) // overlay-only, extends a base word
	weighted.AddWeighted("กา", 50)    // in both: shadows the base weight
	weighted.AddWeighted("อาริน", 99) // overlay-only, new branch
	unweighted := dict.NewTrie()
	for _, w := range overlayWords {
		unweighted.Add(w)
	}

	// ก(0)า(1)แ(2)ฟ(3)ด(4)ำ(5)อ(6)า(7)ร(8)ิ(9)น(10)แ(11)ฟ(12)น(13)
	text := []rune("กาแฟดำอารินแฟน")

	ow := NewOverlayDict(base, weighted)
	checkPrefixWeights(t, "weighted/both+ext", ow, text, 0, []int32{2, 4, 6}, []int32{50, 20, 7})
	checkPrefixWeights(t, "weighted/overlay-only", ow, text, 6, []int32{5}, []int32{99})
	checkPrefixWeights(t, "weighted/base-only", ow, text, 11, []int32{3}, []int32{5})
	checkPrefixWeights(t, "weighted/no-hit", ow, text, 5, []int32{}, []int32{})

	ou := NewOverlayDict(base, unweighted)
	checkPrefixWeights(t, "unweighted/base-weight-kept", ou, text, 0, []int32{2, 4, 6}, []int32{10, 20, 0})
	checkPrefixWeights(t, "unweighted/overlay-only", ou, text, 6, []int32{5}, []int32{0})
	checkPrefixWeights(t, "unweighted/base-only", ou, text, 11, []int32{3}, []int32{5})

	// empty overlay: pure pass-through of the base result
	oe := NewOverlayDict(base, dict.NewTrie())
	checkPrefixWeights(t, "empty/base", oe, text, 0, []int32{2, 4}, []int32{10, 20})

	// unweighted base + weighted overlay: overlay weights still work
	ub := dict.NewTrie()
	ub.Add("กา")
	ub.Add("กาแฟ")
	ouw := NewOverlayDict(weightedFlat(t, ub), weighted)
	checkPrefixWeights(t, "unweighted-base", ouw, text, 0, []int32{2, 4, 6}, []int32{50, 0, 7})
}

// TestOverlayPrefixWeightsPrefixerOnlyBase: a base that implements only
// dict.Prefixer contributes its lengths with weight 0; overlay weights work.
func TestOverlayPrefixWeightsPrefixerOnlyBase(t *testing.T) {
	baseTr := dict.NewTrie()
	baseTr.AddWeighted("กา", 10) // weight invisible through Prefixer
	baseTr.Add("กาแฟ")
	ov := dict.NewTrie()
	ov.AddWeighted("กา", 50)
	ov.AddWeighted("กาแฟดำ", 7)
	o := NewOverlayDict(prefixerOnly{baseTr}, ov)
	text := []rune("กาแฟดำ")
	checkPrefixWeights(t, "prefixer-only", o, text, 0, []int32{2, 4, 6}, []int32{50, 0, 7})
	checkPrefixWeights(t, "prefixer-only/no-hit", o, text, 2, []int32{}, []int32{})
}

// TestOverlayPrefixWeightsMatchesPrefixLens: property — PrefixWeights must
// report exactly the lengths PrefixLens does, at every position, against the
// real shared base dictionary.
func TestOverlayPrefixWeightsMatchesPrefixLens(t *testing.T) {
	base, err := dict.Default()
	if err != nil {
		t.Fatal(err)
	}
	ov := dict.NewTrie()
	ov.AddWeighted("รักภาษา", 12)
	ov.AddWeighted("ภาษา", 3) // also a base word
	o := NewOverlayDict(base, ov)
	text := []rune("ผมชื่อรักภาษาครับ abc ภาษาไทย")
	var lensCopy []int
	var outL, outW []int32
	for s := 0; s < len(text); s++ {
		lensCopy = append(lensCopy[:0], o.PrefixLens(text, s, nil)...) // own the scratch-backed result
		outL, outW = o.PrefixWeights(text, s, outL, outW)
		if len(outL) != len(lensCopy) || len(outW) != len(lensCopy) {
			t.Fatalf("at %d: PrefixLens %d entries, PrefixWeights %d/%d", s, len(lensCopy), len(outL), len(outW))
		}
		for k, l := range lensCopy {
			if int(outL[k]) != l {
				t.Fatalf("at %d: lengths diverge: %v vs %v", s, lensCopy, outL)
			}
		}
	}
	// the overlay word must carry its weight where it matches
	start := 6 // "รักภาษา..." begins after "ผมชื่อ"
	outL, outW = o.PrefixWeights(text, start, outL, outW)
	found := false
	for k, l := range outL {
		if string(text[start:start+int(l)]) == "รักภาษา" {
			found = true
			if outW[k] != 12 {
				t.Errorf("overlay word weight = %d, want 12", outW[k])
			}
		}
	}
	if !found {
		t.Error("overlay word not reported by PrefixWeights")
	}
}
