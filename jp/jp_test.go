package jp

import (
	"bufio"
	"os"
	"strings"
	"sync"
	"testing"
)

func readLines(t testing.TB, path string) []string {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer f.Close()
	var lines []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<24)
	for sc.Scan() {
		lines = append(lines, sc.Text())
	}
	return lines
}

// TestCutCases pins deterministic longest-match output on clear cases.
func TestCutCases(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"コンピュータ", []string{"コンピュータ"}}, // single dictionary word
		{"機械学習", []string{"機械", "学習"}}, // compound → parts (both still indexed)
		{"", nil},
		{"水を飲む", []string{"水", "を", "飲む"}},
		{"ABC 123 と", []string{"ABC", "123", "と"}}, // non-JP runs + spaces dropped
	}
	for _, c := range cases {
		got := Cut(c.in)
		if strings.Join(got, "|") != strings.Join(c.want, "|") {
			t.Errorf("Cut(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

// TestRecallVsSudachi measures how many of Sudachi's multi-char words our
// longest-match segmentation also produces (retrieval-relevant signal). We do
// not require exact parity (Sudachi uses a cost model; we use longest-match).
func TestRecallVsSudachi(t *testing.T) {
	texts := readLines(t, "testdata/ja.txt")
	gold := readLines(t, "testdata/ja_sudachi.golden")
	if len(texts) != len(gold) {
		t.Fatalf("count mismatch")
	}
	var total, hit int
	for i, txt := range texts {
		ours := map[string]bool{}
		for _, w := range Cut(txt) {
			ours[w] = true
		}
		for _, w := range strings.Split(gold[i], "|") {
			if len([]rune(w)) < 2 {
				continue
			}
			total++
			if ours[w] {
				hit++
			}
		}
	}
	recall := float64(hit) / float64(total)
	// Longest-match splits some compounds Sudachi keeps whole (e.g. 機械学習 →
	// 機械|学習); for BM25 indexing the parts are still indexed, so exact parity
	// with Sudachi's C-mode understates usefulness. We just guard a sane floor.
	t.Logf("multi-char word recall vs Sudachi: %d/%d = %.1f%%", hit, total, 100*recall)
	// Floor set just below the measured actual (66.7% on this set) so any
	// silent regression fails; raise it if the measured recall improves.
	if recall < 0.65 {
		t.Errorf("recall %.2f regressed (< 0.65; measured 0.667)", recall)
	}
}

func TestEmbeddedSizeAndLoad(t *testing.T) {
	if EmbeddedSize() <= 0 {
		t.Fatal("EmbeddedSize should be > 0")
	}
	s, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(s.Cut("機械学習"), "|") != strings.Join(Cut("機械学習"), "|") {
		t.Error("Load() instance differs from shared Cut")
	}
}

func TestCutConcurrent(t *testing.T) {
	var wg sync.WaitGroup
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				if len(Cut("自然言語処理")) == 0 {
					t.Error("empty cut")
					return
				}
			}
		}()
	}
	wg.Wait()
}

func TestAppendBytesMatchesCut(t *testing.T) {
	for _, s := range []string{"機械学習", "水を飲む", "ABC 123 と"} {
		want := strings.Join(Cut(s), " ")
		s2, _ := Default()
		got := string(s2.AppendBytes(nil, s, ' '))
		if got != want {
			t.Errorf("AppendBytes(%q)=%q want %q", s, got, want)
		}
	}
}

// FuzzAppendBytesEqualsCut guards the zero-alloc path against the []string path.
func FuzzAppendBytesEqualsCut(f *testing.F) {
	seg, _ := Default()
	for _, s := range []string{"機械学習", "水を飲む", "ABC 123 と", "", "混ぜ漢字かなカナ"} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		want := strings.Join(seg.Cut(s), " ")
		got := string(seg.AppendBytes(nil, s, ' '))
		if got != want {
			t.Fatalf("mismatch %q: %q vs %q", s, got, want)
		}
	})
}

// TestCutDPValid: DP output must reconstruct the input (minus spaces).
func TestCutDPValid(t *testing.T) {
	for _, s := range []string{"機械学習", "自然言語処理", "日本語を勉強", "東京都に住む"} {
		if strings.Join(CutDP(s), "") != strings.ReplaceAll(s, " ", "") {
			t.Errorf("CutDP(%q)=%v does not reconstruct input", s, CutDP(s))
		}
	}
}
