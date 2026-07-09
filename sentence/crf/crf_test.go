package crf

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

// TestGolden measures accuracy against PyThaiNLP crfcut (exact sentence match).
func TestGolden(t *testing.T) {
	in := readLines(t, "testdata/crfcut.txt")
	gold := readLines(t, "testdata/crfcut.golden")
	if len(in) != len(gold) {
		t.Fatalf("count mismatch: in=%d gold=%d", len(in), len(gold))
	}
	diffs := 0
	for i := range in {
		got := strings.Join(Split(in[i]), "|")
		if got != gold[i] {
			diffs++
			if diffs <= 12 {
				t.Errorf("line %d\n  in  =%q\n  got =%q\n  want=%q", i+1, in[i], got, gold[i])
			}
		}
	}
	acc := 100 * float64(len(in)-diffs) / float64(len(in))
	t.Logf("crfcut parity: %d/%d exact (%.2f%%), %d diffs", len(in)-diffs, len(in), acc, diffs)
	if diffs > 0 {
		t.Fail()
	}
}

func TestEmpty(t *testing.T) {
	if got := Split(""); got != nil {
		t.Errorf("Split(\"\") = %v, want nil", got)
	}
}

// TestDialogueModel checks the embedded dialogue-register model loads and
// segments dialogue-heavy text at quote/punctuation boundaries where the default
// (TED-trained) model does not.
func TestDialogueModel(t *testing.T) {
	m := Dialogue()
	if m == nil || len(m.feats) == 0 {
		t.Fatal("Dialogue() returned an empty model")
	}
	if got := m.Split(""); got != nil {
		t.Errorf("Dialogue().Split(\"\") = %v, want nil", got)
	}
	// Two dialogue turns: the model should cut after the closing quote.
	text := "“ปล่อยฉันไปนะ” เธอตะโกน “ฉันจ่ายเงินให้แล้ว”"
	got := m.Split(text)
	if len(got) < 2 {
		t.Errorf("Dialogue().Split gave %d segment(s), want >=2: %q", len(got), got)
	}
	// Concurrent reads must be safe (shared read-only model).
	done := make(chan struct{})
	for i := 0; i < 4; i++ {
		go func() { _ = m.Split(text); done <- struct{}{} }()
	}
	for i := 0; i < 4; i++ {
		<-done
	}
}
