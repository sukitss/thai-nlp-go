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
	if recall < 0.75 {
		t.Errorf("recall %.2f too low (< 0.75)", recall)
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
