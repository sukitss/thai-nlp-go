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
