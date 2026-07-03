package cjk

import (
	"bufio"
	"os"
	"strings"
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

// TestCutCases pins the segmenter's own (deterministic longest-match) output on
// clear cases where longest-match and a dictionary agree.
func TestCutCases(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"北京大学", []string{"北京大学"}}, // one dictionary word
		{"我爱自然语言处理", []string{"我", "爱", "自然语言", "处理"}},
		{"中文分词测试", []string{"中文", "分词", "测试"}},
		{"", nil},
		{"AT&T很酷", []string{"AT", "&", "T", "很酷"}},             // non-Han runs + Han
		{"深度学习 和 机器学习", []string{"深度", "学习", "和", "机器", "学习"}}, // longest-match; spaces dropped
	}
	for _, c := range cases {
		got := Cut(c.in)
		if strings.Join(got, "|") != strings.Join(c.want, "|") {
			t.Errorf("Cut(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

// TestRecallVsJieba measures how many of jieba's multi-character words our
// longest-match segmentation also produces — the retrieval-relevant signal.
// (We don't require exact parity: jieba uses DAG+HMM; we use longest-match.)
func TestRecallVsJieba(t *testing.T) {
	texts := readLines(t, "testdata/zh.txt")
	gold := readLines(t, "testdata/zh_jieba.golden")
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
			if len([]rune(w)) < 2 { // only multi-char words matter for recall
				continue
			}
			total++
			if ours[w] {
				hit++
			}
		}
	}
	recall := float64(hit) / float64(total)
	t.Logf("multi-char word recall vs jieba: %d/%d = %.1f%%", hit, total, 100*recall)
	// Floor set just below the measured actual (85.7% on this set) so any
	// silent regression fails; raise it if the measured recall improves.
	if recall < 0.85 {
		t.Errorf("recall %.2f regressed (< 0.85; measured 0.857)", recall)
	}
}

func TestCutConcurrent(t *testing.T) {
	done := make(chan bool, 4)
	for g := 0; g < 4; g++ {
		go func() {
			for i := 0; i < 200; i++ {
				if len(Cut("自然语言处理")) == 0 {
					t.Error("empty cut")
					break
				}
			}
			done <- true
		}()
	}
	for g := 0; g < 4; g++ {
		<-done
	}
}

func TestLoadOwnInstance(t *testing.T) {
	if EmbeddedSize() <= 0 {
		t.Fatal("EmbeddedSize should be > 0")
	}
	s, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	// an explicitly loaded instance segments the same as the shared one
	if strings.Join(s.Cut("自然语言处理"), "|") != strings.Join(Cut("自然语言处理"), "|") {
		t.Error("Load() instance differs from shared Cut")
	}
}

func TestAppendBytesMatchesCut(t *testing.T) {
	for _, s := range []string{"我爱自然语言处理", "AT&T很酷 深度学习", "北京大学"} {
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
	for _, s := range []string{"我爱自然语言处理", "AT&T 深度学习", "北京大学", "", "混ぜabc123"} {
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

// TestCutDPBeatsGreedy: on a jieba reference set, the frequency DAG+DP path
// should reach higher recall than greedy longest-match (freq disambiguation
// pays off) — and never lose.
func TestCutDPBeatsGreedy(t *testing.T) {
	txt := readLines(t, "testdata/zh_eval.txt")
	gold := readLines(t, "testdata/zh_eval.golden")
	rec := func(cut func(string) []string) float64 {
		var tot, hit int
		for i, s := range txt {
			o := map[string]bool{}
			for _, w := range cut(s) {
				o[w] = true
			}
			for _, w := range strings.Split(gold[i], "|") {
				if len([]rune(w)) < 2 {
					continue
				}
				tot++
				if o[w] {
					hit++
				}
			}
		}
		return float64(hit) / float64(tot)
	}
	greedy, dp := rec(Cut), rec(CutDP)
	t.Logf("recall vs jieba: Cut(greedy)=%.1f%% CutDP(freq)=%.1f%%", 100*greedy, 100*dp)
	if dp < greedy {
		t.Errorf("CutDP (%.3f) should not be worse than Cut (%.3f)", dp, greedy)
	}
	// Floors just below the measured actuals (greedy 86.0%, DP 94.7% on this
	// 18-sentence set) so silent regressions fail; raise them if recall improves.
	if greedy < 0.85 {
		t.Errorf("greedy recall %.3f regressed (< 0.85; measured 0.860)", greedy)
	}
	if dp < 0.94 {
		t.Errorf("DP recall %.3f regressed (< 0.94; measured 0.947)", dp)
	}
}

// TestCutDPValid: DP output must cover the input exactly (no lost/extra runes).
func TestCutDPValid(t *testing.T) {
	for _, s := range []string{"北京大学生", "我爱自然语言处理", "AT&T很酷", "机器学习和深度学习"} {
		if strings.Join(CutDP(s), "") != strings.ReplaceAll(s, " ", "") {
			t.Errorf("CutDP(%q)=%v does not reconstruct input", s, CutDP(s))
		}
	}
}
