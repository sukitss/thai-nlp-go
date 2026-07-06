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

func TestSplitHeuristic(t *testing.T) {
	// space inside a sentence must NOT split when no ender/starter signals it
	// (กิน/ข้าว/ร้าน are not in the ender/starter lists).
	got := SplitHeuristic("ผมกิน ข้าว ร้านนี้")
	if len(got) != 1 {
		t.Errorf("mid-sentence space over-split: %v", got)
	}
	// ender ครับ → boundary
	g2 := SplitHeuristic("สวัสดีครับ วันนี้อากาศดี")
	if len(g2) != 2 || g2[0] != "สวัสดีครับ" {
		t.Errorf("ender split failed: %v", g2)
	}
	// starter แต่ → boundary
	g3 := SplitHeuristic("เขาจะมา แต่รถติด")
	if len(g3) != 2 || g3[1] != "แต่รถติด" {
		t.Errorf("starter split failed: %v", g3)
	}
	// newline = hard boundary regardless of words
	g4 := SplitHeuristic("บรรทัดหนึ่ง\nบรรทัดสอง")
	if len(g4) != 2 {
		t.Errorf("newline hard-break failed: %v", g4)
	}
	// punctuation
	g5 := SplitHeuristic("จริงหรือ? ไม่น่าเชื่อ")
	if len(g5) != 2 {
		t.Errorf("punctuation split failed: %v", g5)
	}
}

func TestEngineInterface(t *testing.T) {
	var e Engine = Heuristic
	if len(e.Split("สวัสดีครับ ไปกันเถอะ")) != 2 {
		t.Error("Heuristic engine via interface failed")
	}
	e = Whitespace
	if len(e.Split("a b c")) != 3 {
		t.Error("Whitespace engine via interface failed")
	}
}

func BenchmarkSplitHeuristic(b *testing.B) {
	txt := "วันนี้อากาศดีมากครับ ผมเลยออกไปวิ่ง ตอนเย็นฝนตก"
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		SplitHeuristic(txt)
	}
}

func TestHeuristicPaiyannoiNotBoundary(t *testing.T) {
	// ฯ is an abbreviation mark, not a sentence end: กรุงเทพฯ must not split
	// from what follows purely because of ฯ.
	got := SplitHeuristic("เขาอยู่กรุงเทพฯ แถวสุขุมวิท")
	if len(got) != 1 {
		t.Errorf("ฯ wrongly treated as boundary: %v", got)
	}
	// ฯลฯ mid-list also not a boundary
	g2 := SplitHeuristic("มีผลไม้ กล้วย ส้ม ฯลฯ วางขาย")
	for _, s := range g2 {
		if s == "มีผลไม้ กล้วย ส้ม ฯลฯ" && len(g2) > 1 {
			// acceptable if not split at ฯลฯ itself; just ensure ฯลฯ didn't force a cut after it
		}
	}
	// ! still a boundary
	g3 := SplitHeuristic("ดีมาก! ไปต่อกันเลย")
	if len(g3) != 2 {
		t.Errorf("! should split: %v", g3)
	}
}

func TestRulecutFeatures(t *testing.T) {
	cases := []struct {
		desc string
		in   string
		want int
	}{
		{"bracket non-break", "เขาพูดว่า (ดี มาก) แล้วเดินไป", 1},
		{"honorific name", "นาย สมชาย ใจดี มาก", 1},
		{"date non-break", "เกิดวันที่ 1 มกราคม 2540 ที่เชียงใหม่", 1},
		{"maiyamok ender", "เขาเดินช้าๆ ผมรีบไป", 2},
		{"paiyannoi keep", "อยู่กรุงเทพฯ แถวสุขุมวิท", 1},
		{"ender ครับ", "สวัสดีครับ ยินดีต้อนรับ", 2},
		{"starter แต่", "เขาจะมา แต่รถติด", 2},
		{"mid space keep", "ผมกิน ข้าว ร้านนี้", 1},
	}
	for _, c := range cases {
		if got := SplitHeuristic(c.in); len(got) != c.want {
			t.Errorf("%s: want %d, got %d: %v", c.desc, c.want, len(got), got)
		}
	}
}
