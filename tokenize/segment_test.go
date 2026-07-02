package tokenize

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
	if err := sc.Err(); err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return lines
}

func defaultSeg(t testing.TB) *Segmenter {
	t.Helper()
	seg, err := NewDefault()
	if err != nil {
		t.Fatalf("NewDefault: %v", err)
	}
	return seg
}

// goldenCase runs SegmentNoWS over an input file and compares the space-joined
// output to a golden file produced by PyThaiNLP
// word_tokenize(engine="newmm", keep_whitespace=False).
func goldenCase(t *testing.T, inPath, goldPath string) {
	seg := defaultSeg(t)
	in := readLines(t, inPath)
	gold := readLines(t, goldPath)
	if len(in) != len(gold) {
		t.Fatalf("line count mismatch: %s=%d %s=%d", inPath, len(in), goldPath, len(gold))
	}
	diffs := 0
	for i := range in {
		got := strings.Join(seg.SegmentNoWS(in[i]), " ")
		if got != gold[i] {
			diffs++
			if diffs <= 10 {
				t.Errorf("line %d\n  in   =%q\n  got  =%q\n  want =%q", i+1, in[i], got, gold[i])
			}
		}
	}
	if diffs == 0 {
		t.Logf("✅ %s: %d lines match PyThaiNLP exactly", inPath, len(in))
	} else {
		t.Fatalf("%d/%d lines differ from PyThaiNLP", diffs, len(in))
	}
}

func TestGoldenCorpus(t *testing.T) { goldenCase(t, "testdata/corpus.txt", "testdata/corpus.golden") }
func TestGoldenStress(t *testing.T) { goldenCase(t, "testdata/stress.txt", "testdata/stress.golden") }

// TestSegmentBytesMatchesJoin verifies the zero-alloc []byte path produces
// exactly the same bytes as strings.Join(SegmentNoWS, " ").
func TestSegmentBytesMatchesJoin(t *testing.T) {
	seg := defaultSeg(t)
	for _, line := range readLines(t, "testdata/corpus.txt") {
		want := strings.Join(seg.SegmentNoWS(line), " ")
		got := string(seg.SegmentBytes(line, ' '))
		if got != want {
			t.Fatalf("SegmentBytes mismatch\n  in  =%q\n  got =%q\n  want=%q", line, got, want)
		}
	}
}

// TestAppendBytesReuse verifies AppendBytes appends onto an existing buffer.
func TestAppendBytesReuse(t *testing.T) {
	seg := defaultSeg(t)
	dst := []byte("PREFIX:")
	dst = seg.AppendBytes(dst, "ฉันรักภาษาไทย", ' ')
	got := string(dst)
	if !strings.HasPrefix(got, "PREFIX:") {
		t.Fatalf("AppendBytes overwrote prefix: %q", got)
	}
	rest := strings.TrimPrefix(got, "PREFIX:")
	want := strings.Join(seg.SegmentNoWS("ฉันรักภาษาไทย"), " ")
	if rest != want {
		t.Fatalf("AppendBytes body = %q, want %q", rest, want)
	}
}

func TestEmptyInput(t *testing.T) {
	seg := defaultSeg(t)
	if got := seg.Segment(""); got != nil {
		t.Errorf("Segment(\"\") = %v, want nil", got)
	}
	if got := seg.SegmentNoWS(""); got != nil {
		t.Errorf("SegmentNoWS(\"\") = %v, want nil", got)
	}
	if got := seg.SegmentBytes("", ' '); len(got) != 0 {
		t.Errorf("SegmentBytes(\"\") = %v, want empty", got)
	}
}

// TestKeepWhitespace checks that Segment retains whitespace tokens while
// SegmentNoWS drops them.
func TestKeepWhitespace(t *testing.T) {
	seg := defaultSeg(t)
	raw := seg.Segment("ฉัน รัก")
	hasWS := false
	for _, tok := range raw {
		if strings.TrimSpace(tok) == "" {
			hasWS = true
		}
	}
	if !hasWS {
		t.Errorf("Segment dropped whitespace: %q", raw)
	}
	for _, tok := range seg.SegmentNoWS("ฉัน รัก") {
		if strings.TrimSpace(tok) == "" {
			t.Errorf("SegmentNoWS kept a blank token: %q", raw)
		}
	}
}
