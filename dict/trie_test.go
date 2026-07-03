package dict

import (
	"bytes"
	"maps"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestTrieLen(t *testing.T) {
	tr := NewTrie()
	if tr.Len() != 0 {
		t.Fatalf("empty Len = %d", tr.Len())
	}
	tr.Add("กา")
	tr.Add("กา") // duplicate not recounted
	tr.Add("  ") // blank ignored
	tr.Add("")
	tr.AddWeighted("กาแฟ", 5)
	tr.AddWeighted("กาแฟ", 6) // duplicate not recounted
	tr.AddWeighted(" ", 1)    // blank ignored
	if tr.Len() != 2 {
		t.Fatalf("Len = %d, want 2", tr.Len())
	}
}

func TestLoadDict(t *testing.T) {
	if _, err := LoadDict(filepath.Join(t.TempDir(), "missing.txt")); err == nil {
		t.Fatal("LoadDict(nonexistent) = nil error")
	}
	path := filepath.Join(t.TempDir(), "d.txt")
	if err := os.WriteFile(path, []byte("กา\n\nกาแฟ\r\nab \n"), 0o644); err != nil {
		t.Fatal(err)
	}
	tr, err := LoadDict(path)
	if err != nil {
		t.Fatal(err)
	}
	if tr.Len() != 3 {
		t.Fatalf("Len = %d, want 3", tr.Len())
	}
	if got := tr.PrefixLens([]rune("กาแฟ"), 0, nil); len(got) != 2 || got[0] != 2 || got[1] != 4 {
		t.Fatalf("PrefixLens = %v, want [2 4]", got)
	}
}

// TestLoadDictWeightedParsing pins the strict weight parser: only base-10
// int32 weights are accepted; malformed-weight lines are skipped entirely.
func TestLoadDictWeightedParsing(t *testing.T) {
	if _, err := LoadDictWeighted(filepath.Join(t.TempDir(), "missing.txt")); err == nil {
		t.Fatal("LoadDictWeighted(nonexistent) = nil error")
	}
	lines := "" +
		"ok\t10\n" +
		"neg\t-5\n" +
		"crlf\t7\r\n" + // \r after weight must not break parsing
		"zero\t0\n" +
		"max\t2147483647\n" +
		"min\t-2147483648\n" +
		"notab\n" + // no tab: added unweighted
		"garbage\t12-3\n" + // skipped (was silently parsed as 123)
		"overflow\t99999999999\n" + // skipped (was silently truncated)
		"trailing\t5x9\n" + // skipped (was parsed as 5)
		"empty\t\n" + // skipped (was weight 0)
		"\n"
	path := filepath.Join(t.TempDir(), "w.txt")
	if err := os.WriteFile(path, []byte(lines), 0o644); err != nil {
		t.Fatal(err)
	}
	tr, err := LoadDictWeighted(path)
	if err != nil {
		t.Fatal(err)
	}
	ft, err := FromBytes(buildFlatBytes(t, tr))
	if err != nil {
		t.Fatal(err)
	}
	if !ft.Weighted() {
		t.Fatal("Weighted() = false")
	}
	want := map[string]int32{
		"ok": 10, "neg": -5, "crlf": 7, "zero": 0,
		"max": math.MaxInt32, "min": math.MinInt32, "notab": 0,
	}
	if tr.Len() != len(want) {
		t.Errorf("Len = %d, want %d", tr.Len(), len(want))
	}
	for w, wt := range want {
		got, ok := ft.Weight(w)
		if !ok || got != wt {
			t.Errorf("Weight(%q) = (%d,%v), want (%d,true)", w, got, ok, wt)
		}
	}
	for _, w := range []string{"garbage", "overflow", "trailing", "empty"} {
		if ft.Contains(w) {
			t.Errorf("malformed-weight line %q was added", w)
		}
	}
}

func TestTrieWeightContains(t *testing.T) {
	tr := NewTrie()
	if tr.Weighted() {
		t.Error("empty trie reports Weighted()=true")
	}
	tr.Add(" กา ") // Add trims
	tr.AddWeighted("กาแฟ", 7)
	if !tr.Weighted() {
		t.Error("Weighted() = false after AddWeighted")
	}
	if !tr.Contains("กา") {
		t.Error("Contains(กา) = false")
	}
	if tr.Contains(" กา ") {
		t.Error("lookups must not trim (mirrors FlatTrie)")
	}
	if w, ok := tr.Weight("กาแฟ"); !ok || w != 7 {
		t.Errorf("Weight(กาแฟ) = (%d,%v), want (7,true)", w, ok)
	}
	if w, ok := tr.Weight("กา"); !ok || w != 0 {
		t.Errorf("Weight(กา) = (%d,%v), want (0,true)", w, ok)
	}
	if _, ok := tr.Weight("กาแ"); ok {
		t.Error("internal node reported as word")
	}
	for _, w := range []string{"", "ข", "กาแฟดำ"} {
		if tr.Contains(w) {
			t.Errorf("Contains(%q) = true", w)
		}
	}
}

// TestTriePrefixWeightsMatchesPrefixLens: on the pointer trie, PrefixWeights
// must report exactly the lengths PrefixLens does, with the stored weights.
func TestTriePrefixWeightsMatchesPrefixLens(t *testing.T) {
	tr := NewTrie()
	ref := map[string]int32{"กา": 10, "กาแฟ": -3, "แฟน": 0, "a": 5, "ab": 6}
	for w, wt := range ref {
		tr.AddWeighted(w, wt)
	}
	tr.Add("ดี") // mixed unweighted word reports 0
	ref["ดี"] = 0
	corpus := []rune("กาแฟนabดีกา z")
	var lens []int
	var outL, outW []int32
	for s := 0; s < len(corpus); s++ {
		lens = tr.PrefixLens(corpus, s, lens)
		outL, outW = tr.PrefixWeights(corpus, s, outL, outW)
		if len(outL) != len(lens) || len(outW) != len(lens) {
			t.Fatalf("at %d: %d lens, %d/%d weighted", s, len(lens), len(outL), len(outW))
		}
		for k, l := range lens {
			word := string(corpus[s : s+l])
			if int(outL[k]) != l || outW[k] != ref[word] {
				t.Fatalf("at %d: (%d,%d), want (%d,%d) for %q", s, outL[k], outW[k], l, ref[word], word)
			}
		}
	}
}

func TestTrieRemove(t *testing.T) {
	cases := []struct {
		name   string
		add    []string
		remove string
		want   bool
		left   []string
	}{
		{"absent word", []string{"กา"}, "กาแฟ", false, []string{"กา"}},
		{"absent internal node", []string{"กาแฟ"}, "กา", false, []string{"กาแฟ"}},
		{"absent sibling branch", []string{"กา"}, "ขา", false, []string{"กา"}},
		{"leaf", []string{"กา", "ขนม"}, "ขนม", true, []string{"กา"}},
		{"prefix of longer word", []string{"กา", "กาแฟ"}, "กา", true, []string{"กาแฟ"}},
		{"word with extensions", []string{"กา", "กาแฟ", "กาแฟดำ"}, "กาแฟ", true, []string{"กา", "กาแฟดำ"}},
		{"extension keeps prefix", []string{"กา", "กาแฟ"}, "กาแฟ", true, []string{"กา"}},
		{"last word", []string{"ก"}, "ก", true, nil},
		{"trims whitespace", []string{"กา"}, " กา ", true, nil},
		{"empty", []string{"กา"}, "", false, []string{"กา"}},
		{"blank", []string{"กา"}, "   ", false, []string{"กา"}},
		{"shared-prefix sibling survives", []string{"abc", "abd"}, "abc", true, []string{"abd"}},
		{"single rune among longer", []string{"a", "ab", "abc"}, "a", true, []string{"ab", "abc"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tr := NewTrie()
			for _, w := range tc.add {
				tr.Add(w)
			}
			if got := tr.Remove(tc.remove); got != tc.want {
				t.Fatalf("Remove(%q) = %v, want %v", tc.remove, got, tc.want)
			}
			if tr.Len() != len(tc.left) {
				t.Fatalf("Len = %d, want %d", tr.Len(), len(tc.left))
			}
			trimmed := strings.TrimSpace(tc.remove)
			if tr.Contains(trimmed) != slices.Contains(tc.left, trimmed) {
				t.Errorf("Contains(%q) = %v after Remove", trimmed, tr.Contains(trimmed))
			}
			for _, w := range tc.left {
				if !tr.Contains(w) {
					t.Errorf("survivor %q lost", w)
				}
			}
			// pruning check: the trie must serialize byte-identically to a
			// fresh build of the surviving words (flatten is deterministic)
			fresh := NewTrie()
			for _, w := range tc.left {
				fresh.Add(w)
			}
			got, err := FlatBytes(tr)
			if err != nil {
				t.Fatal(err)
			}
			want, err := FlatBytes(fresh)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, want) {
				t.Errorf("trie after Remove not pruned to fresh-build shape (%d vs %d bytes)", len(got), len(want))
			}
		})
	}
}

// TestTrieRemoveWeighted: Remove must clear the weight with the word, and must
// not disturb sibling weights.
func TestTrieRemoveWeighted(t *testing.T) {
	tr := NewTrie()
	tr.AddWeighted("กา", 3)
	tr.AddWeighted("กาแฟ", 7)
	if !tr.Remove("กาแฟ") {
		t.Fatal("Remove(กาแฟ) = false")
	}
	if _, ok := tr.Weight("กาแฟ"); ok {
		t.Error("removed word still has a weight entry")
	}
	if w, ok := tr.Weight("กา"); !ok || w != 3 {
		t.Errorf("sibling weight disturbed: (%d,%v)", w, ok)
	}
	if tr.Remove("กาแฟ") {
		t.Error("second Remove = true")
	}
	tr.Add("กาแฟ") // re-add unweighted: old weight must not resurface
	if w, ok := tr.Weight("กาแฟ"); !ok || w != 0 {
		t.Errorf("weight resurfaced after re-Add: (%d,%v)", w, ok)
	}
	// removing a word that prefixes another must clear its weight in place
	tr2 := NewTrie()
	tr2.AddWeighted("กา", 9)
	tr2.AddWeighted("กาแฟ", 4)
	if !tr2.Remove("กา") {
		t.Fatal("Remove(กา) = false")
	}
	tr2.AddWeighted("กา", 0)
	if w, _ := tr2.Weight("กา"); w != 0 {
		t.Errorf("cleared weight resurfaced: %d", w)
	}
}

// TestTrieRemovePruneProperty: after removing a random subset, the trie must
// serialize byte-identically to a trie built from only the surviving words —
// i.e. Remove fully reclaims dead branches and clears weights.
func TestTrieRemovePruneProperty(t *testing.T) {
	words := uniqueSampleWords(t)
	wt := func(w string) int32 { return int32(len(w)) - 4 }
	tr := NewTrie()
	for _, w := range words {
		tr.AddWeighted(w, wt(w))
	}
	rng := rand.New(rand.NewSource(42))
	removed := make(map[string]bool, len(words))
	for i, w := range words {
		if i == 0 || rng.Intn(2) == 0 { // keep words[0] so both tries stay weighted
			continue
		}
		if !tr.Remove(w) {
			t.Fatalf("Remove(%q) = false for a present word", w)
		}
		removed[w] = true
	}
	fresh := NewTrie()
	for _, w := range words {
		if !removed[w] {
			fresh.AddWeighted(w, wt(w))
		}
	}
	if tr.Len() != fresh.Len() {
		t.Fatalf("Len = %d, want %d", tr.Len(), fresh.Len())
	}
	got, err := FlatBytes(tr)
	if err != nil {
		t.Fatal(err)
	}
	want, err := FlatBytes(fresh)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("removed trie serializes to %d bytes, fresh build to %d — not pruned identically", len(got), len(want))
	}
}

func TestTrieWordsOrderAndCompleteness(t *testing.T) {
	words := uniqueSampleWords(t)
	ref := make(map[string]int32, len(words))
	tr := NewTrie()
	for i, w := range words {
		wt := int32(i%17) - 8
		tr.AddWeighted(w, wt)
		ref[w] = wt
	}
	var got []string
	tr.Words(func(w string, wt int32) bool {
		got = append(got, w)
		if wt != ref[w] {
			t.Errorf("Words(%q) weight = %d, want %d", w, wt, ref[w])
		}
		return true
	})
	// UTF-8 byte order equals code-point order, so sorted strings give the
	// expected lexicographic rune order.
	want := slices.Sorted(maps.Keys(ref))
	if !slices.Equal(got, want) {
		t.Fatalf("Words yielded %d words in wrong order/set (want %d)", len(got), len(want))
	}
	if len(got) != tr.Len() {
		t.Errorf("Words yielded %d, Len = %d", len(got), tr.Len())
	}

	// unweighted tries pass weight 0
	un := NewTrie()
	un.Add("กาแฟ")
	un.Add("abc")
	un.Words(func(w string, wt int32) bool {
		if wt != 0 {
			t.Errorf("unweighted Words(%q) weight = %d", w, wt)
		}
		return true
	})

	// fn returning false stops the walk immediately
	calls := 0
	tr.Words(func(string, int32) bool { calls++; return calls < 5 })
	if calls != 5 {
		t.Errorf("early stop made %d calls, want 5", calls)
	}

	NewTrie().Words(func(string, int32) bool { t.Error("fn called on empty trie"); return true })
}

type prefixWalker interface {
	WalkPrefix(prefix string, fn func(word string, weight int32) bool)
}

func TestWalkPrefix(t *testing.T) {
	ref := map[string]int32{
		"ก": 1, "กา": 2, "กาแฟ": 3, "กาชาด": 4, "การบ้าน": 5, "ขนม": 6,
		"a": 7, "ab": 8, "abc": 9, "b": 10, "๑": 11, "๑๒๓": 12, "กาแฟดำเย็น": 13,
	}
	tr := NewTrie()
	for w, wt := range ref {
		tr.AddWeighted(w, wt)
	}
	data, err := FlatBytes(tr)
	if err != nil {
		t.Fatal(err)
	}
	ft, err := FromBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	sorted := slices.Sorted(maps.Keys(ref))
	prefixes := []string{
		"",           // all words
		"ก",          // itself a word, many extensions
		"กา",         // itself a word
		"กาแ",        // internal node only
		"กาแฟดำเย็น", // itself a word, no extensions
		"กาแฟดำเย็นจัด", // longer than any word
		"ข", "a", "ab", "๑๒",
		"z", "ฮ", // not in dict
	}
	for name, d := range map[string]prefixWalker{"Trie": tr, "FlatTrie": ft} {
		for _, p := range prefixes {
			var want []string
			for _, w := range sorted {
				if strings.HasPrefix(w, p) {
					want = append(want, w)
				}
			}
			var got []string
			d.WalkPrefix(p, func(w string, wt int32) bool {
				got = append(got, w)
				if wt != ref[w] {
					t.Errorf("%s: WalkPrefix(%q) weight for %q = %d, want %d", name, p, w, wt, ref[w])
				}
				return true
			})
			if !slices.Equal(got, want) {
				t.Errorf("%s: WalkPrefix(%q) = %v, want %v", name, p, got, want)
			}
		}
		calls := 0
		d.WalkPrefix("", func(string, int32) bool { calls++; return calls < 3 })
		if calls != 3 {
			t.Errorf("%s: early stop made %d calls, want 3", name, calls)
		}
		calls = 0
		d.WalkPrefix("กา", func(string, int32) bool { calls++; return false }) // stop on the prefix word itself
		if calls != 1 {
			t.Errorf("%s: stop-at-first made %d calls, want 1", name, calls)
		}
	}
}
