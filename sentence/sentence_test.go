package sentence

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

func goldenCase(t *testing.T, goldFile string, fn func(string) []string) {
	in := readLines(t, "testdata/sent.txt")
	gold := readLines(t, goldFile)
	if len(in) != len(gold) {
		t.Fatalf("count mismatch: in=%d gold=%d", len(in), len(gold))
	}
	diffs := 0
	for i := range in {
		got := strings.Join(fn(in[i]), "|")
		if got != gold[i] {
			diffs++
			if diffs <= 10 {
				t.Errorf("line %d in=%q got=%q want=%q", i+1, in[i], got, gold[i])
			}
		}
	}
	if diffs == 0 {
		t.Logf("✅ %s: %d lines match PyThaiNLP", goldFile, len(in))
	}
}

func TestSplitFieldsGolden(t *testing.T) {
	goldenCase(t, "testdata/sent_fields.golden", Split)
}

func TestSplitSpacesGolden(t *testing.T) {
	goldenCase(t, "testdata/sent_ws.golden", SplitSpaces)
}

func TestEmpty(t *testing.T) {
	if got := Split(""); len(got) != 0 {
		t.Errorf("Split(\"\") = %v, want empty", got)
	}
	if got := SplitSpaces(""); got != nil {
		t.Errorf("SplitSpaces(\"\") = %v, want nil", got)
	}
}
