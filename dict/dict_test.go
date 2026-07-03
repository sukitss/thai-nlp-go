package dict

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// TestDefaultIsShared verifies Default returns the same instance every time —
// the whole point is loading the dictionary once and sharing it (perf/mem).
func TestDefaultIsShared(t *testing.T) {
	a, err := Default()
	if err != nil {
		t.Fatalf("Default: %v", err)
	}
	b, err := Default()
	if err != nil {
		t.Fatalf("Default (2nd): %v", err)
	}
	if a != b {
		t.Fatalf("Default returned different instances: %p vs %p", a, b)
	}
}

// TestDefaultLookup sanity-checks that a known word is found.
func TestDefaultLookup(t *testing.T) {
	d, err := Default()
	if err != nil {
		t.Fatalf("Default: %v", err)
	}
	text := []rune("ภาษาไทย")
	lens := d.PrefixLens(text, 0, nil)
	if len(lens) == 0 {
		t.Fatalf("expected some dictionary prefix for %q, got none", string(text))
	}
}

// sampleWords returns a few hundred words from the embedded word list plus a
// crafted diverse set (Thai/Latin/digits/single-rune/long/shared prefixes).
func sampleWords(t *testing.T) []string {
	t.Helper()
	words := []string{
		"ก", "กา", "กาแฟ", "แฟน", "a", "ab", "abc", "abcdef",
		"1", "12", "123456789", "ๆ", "กรุงเทพมหานคร", "กรุง", "เทพ",
	}
	f, err := os.Open(filepath.Join("data", "words_th.txt"))
	if err != nil {
		t.Fatalf("open word list: %v", err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	const stride = 200 // ~62k words / 200 ≈ 310 samples
	for i := 0; sc.Scan(); i++ {
		if i%stride == 0 {
			if w := strings.TrimSpace(sc.Text()); w != "" {
				words = append(words, w)
			}
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if len(words) < 200 {
		t.Fatalf("sampled only %d words", len(words))
	}
	return words
}

// TestFlatMatchesPointer verifies the flat trie is lookup-equivalent to the
// pointer trie it was built from, across all three load paths.
func TestFlatMatchesPointer(t *testing.T) {
	words := sampleWords(t)
	pt := NewTrie()
	inDict := make(map[string]bool, len(words))
	for _, w := range words {
		pt.Add(w)
		inDict[w] = true
	}

	path := filepath.Join(t.TempDir(), "eq.fdt")
	if err := BuildFlatFromTrie(pt, path); err != nil {
		t.Fatal(err)
	}
	mm, err := OpenFlat(path)
	if err != nil {
		t.Fatal(err)
	}
	defer mm.Close()
	eager, err := ReadFlat(path)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	fb, err := FromBytes(raw)
	if err != nil {
		t.Fatal(err)
	}

	corpus := []rune(strings.Join(words, "") + "zzzไม่มีในพจนานุกรมqq")
	negatives := []string{"", "z", "ไม่มีในพจนานุกรมแน่นอน", "abcdefg"}
	for name, ft := range map[string]*FlatTrie{"OpenFlat": mm, "ReadFlat": eager, "FromBytes": fb} {
		if ft.Weighted() {
			t.Errorf("%s: unweighted build reports Weighted()=true", name)
		}
		for _, w := range words {
			if !ft.Contains(w) {
				t.Fatalf("%s: Contains(%q)=false, want true", name, w)
			}
			if wt, ok := ft.Weight(w); !ok || wt != 0 {
				t.Fatalf("%s: Weight(%q)=(%d,%v), want (0,true)", name, w, wt, ok)
			}
		}
		for _, w := range negatives {
			if inDict[w] {
				continue
			}
			if ft.Contains(w) {
				t.Fatalf("%s: Contains(%q)=true, want false", name, w)
			}
		}
		var lens []int
		var outL, outW []int32
		var want []int
		for start := 0; start < len(corpus); start++ {
			want = pt.PrefixLens(corpus, start, want)
			lens = ft.PrefixLens(corpus, start, lens)
			if len(lens) != len(want) {
				t.Fatalf("%s: PrefixLens at %d = %v, want %v", name, start, lens, want)
			}
			outL, outW = ft.PrefixWeights(corpus, start, outL, outW)
			if len(outL) != len(want) || len(outW) != len(want) {
				t.Fatalf("%s: PrefixWeights at %d: %d lens/%d weights, want %d", name, start, len(outL), len(outW), len(want))
			}
			for k := range want {
				if lens[k] != want[k] || int(outL[k]) != want[k] {
					t.Fatalf("%s: at %d got lens=%v prefixWeightLens=%v, want %v", name, start, lens, outL, want)
				}
				if outW[k] != 0 {
					t.Fatalf("%s: unweighted PrefixWeights weight = %d, want 0", name, outW[k])
				}
			}
		}
	}
}

// TestConcurrentDefaultReads hammers the shared Default() trie from many
// goroutines, including racing the first-call load path (run under -race).
func TestConcurrentDefaultReads(t *testing.T) {
	const goroutines = 32
	text := []rune("ผมชอบกินข้าวผัดกับไข่ดาวมากๆเลยครับ")
	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			d, err := Default()
			if err != nil {
				t.Error(err)
				return
			}
			lens := make([]int, 0, 8)
			outL, outW := make([]int32, 0, 8), make([]int32, 0, 8)
			for iter := 0; iter < 50; iter++ {
				for s := 0; s < len(text); s++ {
					lens = d.PrefixLens(text, s, lens)
					outL, outW = d.PrefixWeights(text, s, outL, outW)
					if len(lens) != len(outL) {
						t.Errorf("PrefixLens/PrefixWeights disagree at %d", s)
						return
					}
				}
				if !d.Contains("ภาษาไทย") {
					t.Error("Contains(ภาษาไทย)=false")
					return
				}
				if _, ok := d.Weight("การ"); !ok {
					t.Error("Weight(การ) not found")
					return
				}
			}
		}()
	}
	wg.Wait()
}

func TestWeightAndContains(t *testing.T) {
	d, err := Default()
	if err != nil {
		t.Fatal(err)
	}
	if !d.Contains("ภาษาไทย") {
		t.Error("ภาษาไทย should be in the dict")
	}
	if d.Contains("ไม่ใช่คำจริงๆเลย") {
		t.Error("nonsense should not be in the dict")
	}
	// weighted dict: a common word should have a (nonzero) weight
	if d.Weighted() {
		w, ok := d.Weight("การ")
		if !ok {
			t.Error("การ should be present")
		}
		_ = w
	}
	if _, ok := d.Weight(""); ok {
		t.Error("empty string should not be present")
	}
}
