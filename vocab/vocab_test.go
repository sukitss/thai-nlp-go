package vocab

import (
	"bytes"
	"fmt"
	"math"
	"sync"
	"testing"
)

// TestIDsSequentialAndStable: ids are 0,1,2,... in first-seen order, and the
// same insert order always yields the same ids.
func TestIDsSequentialAndStable(t *testing.T) {
	terms := []string{"แมว", "หมา", "cat", "🐱", "", "แมว", "cat", "นก"}
	want := []uint32{0, 1, 2, 3, 4, 0, 2, 5}
	b := NewBuilder()
	for i, term := range terms {
		if got := b.GetOrAssign(term); got != want[i] {
			t.Errorf("GetOrAssign(%q) #%d = %d, want %d", term, i, got, want[i])
		}
	}
	if b.Len() != 6 {
		t.Fatalf("Len = %d, want 6", b.Len())
	}
	// stability: a fresh builder with the same order assigns the same ids
	b2 := NewBuilder()
	for i, term := range terms {
		if got := b2.GetOrAssign(term); got != want[i] {
			t.Errorf("second builder: GetOrAssign(%q) #%d = %d, want %d", term, i, got, want[i])
		}
	}
}

// TestAddDocDF: duplicate terms within one doc count once; GetOrAssign never
// bumps DF or DocCount.
func TestAddDocDF(t *testing.T) {
	b := NewBuilder()
	b.AddDoc([]string{"แมว", "แมว", "แมว", "วิ่ง"}) // แมว repeats → DF 1
	b.AddDoc([]string{"แมว", "นอน"})
	if got := b.DF("แมว"); got != 2 {
		t.Errorf("DF(แมว) = %d, want 2 (dups in one doc count once)", got)
	}
	if got := b.DF("วิ่ง"); got != 1 {
		t.Errorf("DF(วิ่ง) = %d, want 1", got)
	}
	if got := b.DF("ไม่มี"); got != 0 {
		t.Errorf("DF(unknown) = %d, want 0", got)
	}
	if b.DocCount() != 2 {
		t.Fatalf("DocCount = %d, want 2", b.DocCount())
	}
	// GetOrAssign registers the term but leaves DF and DocCount alone
	id := b.GetOrAssign("query-only")
	if got := b.DF("query-only"); got != 0 {
		t.Errorf("DF after GetOrAssign = %d, want 0", got)
	}
	if b.DocCount() != 2 {
		t.Errorf("DocCount changed by GetOrAssign: %d", b.DocCount())
	}
	// ...and a later AddDoc counts it normally with the same id
	b.AddDoc([]string{"query-only"})
	if got, _ := b.ID("query-only"); got != id {
		t.Errorf("id changed after AddDoc: %d != %d", got, id)
	}
	if got := b.DF("query-only"); got != 1 {
		t.Errorf("DF after AddDoc = %d, want 1", got)
	}
}

// TestIDF checks the BM25 formula ln(1+(N-df+0.5)/(df+0.5)) against
// hand-computed values.
func TestIDF(t *testing.T) {
	b := NewBuilder()
	b.AddDoc([]string{"a", "b"})
	b.AddDoc([]string{"a", "b", "c"})
	b.AddDoc([]string{"a", "c"})
	b.AddDoc([]string{"a"})
	// N=4: a df=4, b df=2, c df=2
	cases := []struct {
		term string
		want float64
	}{
		{"a", math.Log(10.0 / 9.0)}, // ln(1+0.5/4.5)            ≈ 0.10536
		{"b", math.Log(2.0)},        // ln(1+2.5/2.5)            ≈ 0.69315
		{"unseen", math.Log(10.0)},  // df=0: ln(1+4.5/0.5)      ≈ 2.30259
		{"c", 0.6931471805599453},   // same as b, literal check
	}
	for _, c := range cases {
		if got := b.IDF(c.term); math.Abs(got-c.want) > 1e-12 {
			t.Errorf("IDF(%q) = %.15f, want %.15f", c.term, got, c.want)
		}
	}
	// empty corpus: every term is unseen with N=0 → ln(1+0.5/0.5) = ln 2
	if got := NewBuilder().IDF("x"); math.Abs(got-math.Ln2) > 1e-12 {
		t.Errorf("empty-corpus IDF = %v, want ln 2", got)
	}
}

func TestTermsIteration(t *testing.T) {
	b := NewBuilder()
	b.AddDoc([]string{"x", "y"})
	b.AddDoc([]string{"y", "z"})
	var gotTerms []string
	var gotIDs []uint32
	var gotDF []int
	b.Terms(func(term string, id uint32, df int) bool {
		gotTerms = append(gotTerms, term)
		gotIDs = append(gotIDs, id)
		gotDF = append(gotDF, df)
		return true
	})
	if fmt.Sprint(gotTerms) != "[x y z]" || fmt.Sprint(gotIDs) != "[0 1 2]" || fmt.Sprint(gotDF) != "[1 2 1]" {
		t.Errorf("Terms = %v %v %v, want [x y z] [0 1 2] [1 2 1]", gotTerms, gotIDs, gotDF)
	}
	// early stop
	n := 0
	b.Terms(func(string, uint32, int) bool { n++; return n < 2 })
	if n != 2 {
		t.Errorf("early stop visited %d terms, want 2", n)
	}
}

// TestExtend: ids and DF survive a Save/Load/Extend cycle, new terms append
// after Len(), and the source Vocab is never modified.
func TestExtend(t *testing.T) {
	b := NewBuilder()
	b.AddDoc([]string{"แมว", "วิ่ง"})
	b.AddDoc([]string{"แมว"})
	v := saveLoad(t, b)

	b2 := v.Extend()
	if b2.Len() != 2 || b2.DocCount() != 2 {
		t.Fatalf("Extend: Len=%d DocCount=%d, want 2/2", b2.Len(), b2.DocCount())
	}
	b2.AddDoc([]string{"แมว", "ใหม่"})
	if id, _ := b2.ID("แมว"); id != 0 {
		t.Errorf("existing id changed after Extend: %d", id)
	}
	if id, _ := b2.ID("ใหม่"); id != 2 {
		t.Errorf("new term id = %d, want 2 (appended after Len)", id)
	}
	if got := b2.DF("แมว"); got != 3 {
		t.Errorf("DF continues across Extend: got %d, want 3", got)
	}
	if b2.DocCount() != 3 {
		t.Errorf("DocCount continues across Extend: got %d, want 3", b2.DocCount())
	}
	// the frozen Vocab is untouched
	if v.Len() != 2 || v.DocCount() != 2 || v.DF("แมว") != 2 {
		t.Errorf("Vocab mutated by Extend+AddDoc: Len=%d DocCount=%d DF=%d", v.Len(), v.DocCount(), v.DF("แมว"))
	}
	if _, ok := v.ID("ใหม่"); ok {
		t.Error("Vocab gained a term added to the extending Builder")
	}
	// extended builder re-saves and round-trips
	v2 := saveLoad(t, b2)
	if v2.Len() != 3 || v2.DF("ใหม่") != 1 {
		t.Errorf("extended round-trip: Len=%d DF(ใหม่)=%d", v2.Len(), v2.DF("ใหม่"))
	}
}

// TestVocabConcurrentReaders hammers a loaded Vocab from many goroutines; run
// with -race to verify reader safety.
func TestVocabConcurrentReaders(t *testing.T) {
	b := NewBuilder()
	for d := 0; d < 50; d++ {
		doc := make([]string, 0, 40)
		for i := d; i < d+40; i++ {
			doc = append(doc, fmt.Sprintf("คำ%d", i))
		}
		b.AddDoc(doc)
	}
	v := saveLoad(t, b)
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			sum := 0.0
			for i := 0; i < 2000; i++ {
				term := fmt.Sprintf("คำ%d", (g*31+i)%100)
				if id, ok := v.ID(term); ok {
					sum += float64(id)
				}
				sum += float64(v.DF(term)) + v.IDF(term)
				if i%500 == 0 {
					v.Terms(func(_ string, id uint32, df int) bool { return id < 10 })
				}
			}
			_ = sum
			if v.DocCount() != 50 {
				t.Errorf("DocCount = %d", v.DocCount())
			}
		}(g)
	}
	wg.Wait()
}

// saveLoad round-trips b through the binary format, failing the test on error.
func saveLoad(t testing.TB, b *Builder) *Vocab {
	t.Helper()
	var buf bytes.Buffer
	if err := b.Save(&buf); err != nil {
		t.Fatalf("Save: %v", err)
	}
	v, err := Load(&buf)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	return v
}
