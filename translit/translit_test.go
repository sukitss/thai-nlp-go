package translit

import (
	"bufio"
	"os"
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

// TestMetasoundGolden checks Metasound matches PyThaiNLP metasound byte-for-byte.
func TestMetasoundGolden(t *testing.T) {
	in := readLines(t, "testdata/meta.txt")
	gold := readLines(t, "testdata/meta.golden")
	if len(in) != len(gold) {
		t.Fatalf("count mismatch: in=%d gold=%d", len(in), len(gold))
	}
	diffs := 0
	for i := range in {
		if got := Metasound(in[i], 4); got != gold[i] {
			diffs++
			if diffs <= 10 {
				t.Errorf("Metasound(%q) = %q, want %q", in[i], got, gold[i])
			}
		}
	}
	if diffs == 0 {
		t.Logf("✅ %d words match PyThaiNLP metasound", len(in))
	}
}

// TestVariantsShareKey checks known spelling variants collide.
func TestVariantsShareKey(t *testing.T) {
	pairs := [][2]string{{"ทองดี", "ทองดา"}, {"บ้าน", "บาน"}, {"รัก", "รักษ์"}}
	for _, p := range pairs {
		if Key(p[0]) != Key(p[1]) {
			t.Errorf("expected same key for %q/%q: %q vs %q", p[0], p[1], Key(p[0]), Key(p[1]))
		}
	}
}

func TestEmpty(t *testing.T) {
	if Key("") != "" {
		t.Errorf("Key(\"\") = %q, want \"\"", Key(""))
	}
	if Metasound("x", 0) != "" {
		t.Errorf("Metasound length 0 should be empty")
	}
}
