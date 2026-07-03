package dict

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestTrieLen(t *testing.T) {
	tr := NewTrie()
	if tr.Len() != 0 {
		t.Fatalf("empty Len = %d", tr.Len())
	}
	tr.Add("กา")
	tr.Add("กา") // duplicate not recounted
	tr.Add("  ") // blank ignored
	tr.Add("")
	tr.AddWeighted("กาแฟ", 5)
	tr.AddWeighted("กาแฟ", 6) // duplicate not recounted
	tr.AddWeighted(" ", 1)    // blank ignored
	if tr.Len() != 2 {
		t.Fatalf("Len = %d, want 2", tr.Len())
	}
}

func TestLoadDict(t *testing.T) {
	if _, err := LoadDict(filepath.Join(t.TempDir(), "missing.txt")); err == nil {
		t.Fatal("LoadDict(nonexistent) = nil error")
	}
	path := filepath.Join(t.TempDir(), "d.txt")
	if err := os.WriteFile(path, []byte("กา\n\nกาแฟ\r\nab \n"), 0o644); err != nil {
		t.Fatal(err)
	}
	tr, err := LoadDict(path)
	if err != nil {
		t.Fatal(err)
	}
	if tr.Len() != 3 {
		t.Fatalf("Len = %d, want 3", tr.Len())
	}
	if got := tr.PrefixLens([]rune("กาแฟ"), 0, nil); len(got) != 2 || got[0] != 2 || got[1] != 4 {
		t.Fatalf("PrefixLens = %v, want [2 4]", got)
	}
}

// TestLoadDictWeightedParsing pins the strict weight parser: only base-10
// int32 weights are accepted; malformed-weight lines are skipped entirely.
func TestLoadDictWeightedParsing(t *testing.T) {
	if _, err := LoadDictWeighted(filepath.Join(t.TempDir(), "missing.txt")); err == nil {
		t.Fatal("LoadDictWeighted(nonexistent) = nil error")
	}
	lines := "" +
		"ok\t10\n" +
		"neg\t-5\n" +
		"crlf\t7\r\n" + // \r after weight must not break parsing
		"zero\t0\n" +
		"max\t2147483647\n" +
		"min\t-2147483648\n" +
		"notab\n" + // no tab: added unweighted
		"garbage\t12-3\n" + // skipped (was silently parsed as 123)
		"overflow\t99999999999\n" + // skipped (was silently truncated)
		"trailing\t5x9\n" + // skipped (was parsed as 5)
		"empty\t\n" + // skipped (was weight 0)
		"\n"
	path := filepath.Join(t.TempDir(), "w.txt")
	if err := os.WriteFile(path, []byte(lines), 0o644); err != nil {
		t.Fatal(err)
	}
	tr, err := LoadDictWeighted(path)
	if err != nil {
		t.Fatal(err)
	}
	ft, err := FromBytes(buildFlatBytes(t, tr))
	if err != nil {
		t.Fatal(err)
	}
	if !ft.Weighted() {
		t.Fatal("Weighted() = false")
	}
	want := map[string]int32{
		"ok": 10, "neg": -5, "crlf": 7, "zero": 0,
		"max": math.MaxInt32, "min": math.MinInt32, "notab": 0,
	}
	if tr.Len() != len(want) {
		t.Errorf("Len = %d, want %d", tr.Len(), len(want))
	}
	for w, wt := range want {
		got, ok := ft.Weight(w)
		if !ok || got != wt {
			t.Errorf("Weight(%q) = (%d,%v), want (%d,true)", w, got, ok, wt)
		}
	}
	for _, w := range []string{"garbage", "overflow", "trailing", "empty"} {
		if ft.Contains(w) {
			t.Errorf("malformed-weight line %q was added", w)
		}
	}
}
