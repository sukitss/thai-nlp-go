package embed

import (
	"math"
	"strings"
	"testing"

	"github.com/sukitss/thai-nlp-go/vocab"
)

// bothImplementEmbedder is a compile-time check that both types satisfy the
// shared interface (so either drops into a vector.Flat the same way).
var (
	_ Embedder = (*Hashing)(nil)
	_ Embedder = (*Static)(nil)
)

func cosine(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("cosine dim mismatch")
	}
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s // both are unit vectors, so dot == cosine
}

func l2(v []float32) float64 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return math.Sqrt(s)
}

// --- Hashing --------------------------------------------------------------

// TestHashingGolden locks the exact output of a fixed input under a fixed
// configuration. If the hashing, n-gram walk, or normalization changes, this
// vector changes — regenerate it deliberately, never silently.
func TestHashingGolden(t *testing.T) {
	h := NewHashing(8, WithNGrams(2, 3))
	got := h.Embed("ปลา IT")
	want := []float32{0, 0, -0.5, 0.25, 0, -0.25, 0.25, 0.75}
	if len(got) != len(want) {
		t.Fatalf("dim = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("golden mismatch at %d: got %v want %v", i, got, want)
		}
	}
}

// TestHashingDeterministic: the same text always yields byte-identical output.
func TestHashingDeterministic(t *testing.T) {
	h := NewHashing(256)
	a := h.Embed("ติดต่อพี่ปลา IT โทร 081-234-5678")
	b := h.Embed("ติดต่อพี่ปลา IT โทร 081-234-5678")
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("non-deterministic at %d: %v vs %v", i, a[i], b[i])
		}
	}
	// A second embedder with the same config must agree.
	if c := NewHashing(256).Embed("ติดต่อพี่ปลา IT โทร 081-234-5678"); c[0] != a[0] {
		t.Fatalf("two embedders disagree: %v vs %v", c[0], a[0])
	}
}

// TestHashingUnitNorm: every non-empty embedding is unit length; empty/
// content-free text embeds to all-zeros.
func TestHashingUnitNorm(t *testing.T) {
	h := NewHashing(128)
	for _, s := range []string{"ปลา", "ขอเบอร์ปลาit หน่อย", "reset password OTP"} {
		if n := l2(h.Embed(s)); math.Abs(n-1) > 1e-5 {
			t.Errorf("‖embed(%q)‖ = %.6f, want 1", s, n)
		}
	}
	for _, s := range []string{"", "   ", "!!!"} {
		if n := l2(h.Embed(s)); n != 0 {
			t.Errorf("‖embed(%q)‖ = %.6f, want 0 (no content)", s, n)
		}
	}
}

// TestHashingVariantSimilarity is the core claim: a spelling/spacing variant
// (stuck acronym, missing space, case) is MORE similar to its true match than to
// an unrelated document — the lexical-fuzzy property BM25 lacks.
func TestHashingVariantSimilarity(t *testing.T) {
	h := NewHashing(512)
	q := h.Embed("ขอเบอร์ปลาit หน่อย") // stuck acronym, lower-case, no space
	rel := h.Embed("ติดต่อพี่ปลา IT โทร 081-234-5678")
	unrel := h.Embed("พิมพ์ใบเสร็จไม่ออก เช็คกระดาษหมึก")
	sRel, sUnrel := cosine(q, rel), cosine(q, unrel)
	if sRel <= sUnrel {
		t.Fatalf("variant not closer to its match: rel=%.4f unrel=%.4f", sRel, sUnrel)
	}
	if sRel < 0.15 {
		t.Errorf("variant similarity unexpectedly low: %.4f", sRel)
	}
}

// TestHashingIDFWeighting: with an IDF vocab, a rare token shifts the vector
// more than a corpus-common one — i.e. the weighting actually changes output.
func TestHashingIDFWeighting(t *testing.T) {
	b := vocab.NewBuilder()
	// "การ" appears in every doc (low IDF); "เน็ตเวิร์ก" in one (high IDF).
	b.AddDoc([]string{"การ", "ระบบ"})
	b.AddDoc([]string{"การ", "รหัส"})
	b.AddDoc([]string{"การ", "เน็ตเวิร์ก"})
	var buf strings.Builder
	if err := b.Save(&buf); err != nil {
		t.Fatal(err)
	}
	v, err := vocab.Load(strings.NewReader(buf.String()))
	if err != nil {
		t.Fatal(err)
	}
	plain := NewHashing(128)
	weighted := NewHashing(128, WithIDF(v))
	if cosine(plain.Embed("การ เน็ตเวิร์ก"), weighted.Embed("การ เน็ตเวิร์ก")) > 0.999 {
		t.Errorf("IDF weighting did not change the embedding")
	}
}

// TestHashingConcurrent: Embed is safe for concurrent callers (run under -race).
func TestHashingConcurrent(t *testing.T) {
	h := NewHashing(256)
	ref := h.Embed("ติดต่อพี่ปลา IT")
	done := make(chan bool)
	for w := 0; w < 8; w++ {
		go func() {
			for i := 0; i < 200; i++ {
				got := h.Embed("ติดต่อพี่ปลา IT")
				if got[0] != ref[0] {
					t.Errorf("concurrent mismatch")
					break
				}
			}
			done <- true
		}()
	}
	for w := 0; w < 8; w++ {
		<-done
	}
}

func TestNewHashingPanicsOnBadDim(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for dim <= 0")
		}
	}()
	NewHashing(0)
}

// --- Static ---------------------------------------------------------------

// tinyVec is a hand-written 3-dim model: two Thai words and one Latin, chosen so
// the pooled result is hand-verifiable.
const tinyVec = `3 3
cat 1.0 0.0 0.0
dog 0.0 1.0 0.0
รถ 0.0 0.0 2.0
`

func loadTiny(t *testing.T, opts ...StaticOption) *Static {
	t.Helper()
	s, err := LoadStatic(strings.NewReader(tinyVec), opts...)
	if err != nil {
		t.Fatalf("LoadStatic: %v", err)
	}
	return s
}

// TestStaticParse: header, dims, exact stored vectors, Len (golden .vec parse).
func TestStaticParse(t *testing.T) {
	s := loadTiny(t)
	if s.Dim() != 3 {
		t.Fatalf("Dim = %d, want 3", s.Dim())
	}
	if s.Len() != 3 {
		t.Fatalf("Len = %d, want 3", s.Len())
	}
	if v, ok := s.Vector("dog"); !ok || v[0] != 0 || v[1] != 1 || v[2] != 0 {
		t.Fatalf("Vector(dog) = %v, %v", v, ok)
	}
	if v, ok := s.Vector("รถ"); !ok || v[2] != 2 {
		t.Fatalf("Vector(รถ) = %v, %v", v, ok)
	}
	if _, ok := s.Vector("missing"); ok {
		t.Fatal("Vector(missing) should be absent")
	}
}

// TestStaticEmbedMeanPool: "cat dog" pools to (0.5,0.5,0)→normalized (1,1,0)/√2.
// An out-of-vocab token is skipped, not counted.
func TestStaticEmbedMeanPool(t *testing.T) {
	s := loadTiny(t)
	got := s.Embed("cat dog zzz") // zzz is OOV → skipped
	inv := float32(1 / math.Sqrt2)
	want := []float32{inv, inv, 0}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("Embed(cat dog zzz) = %v, want %v", got, want)
		}
	}
}

// TestStaticAllOOV: text with no in-vocab token embeds to all-zeros.
func TestStaticAllOOV(t *testing.T) {
	s := loadTiny(t)
	if n := l2(s.Embed("zzz qqq")); n != 0 {
		t.Fatalf("all-OOV norm = %v, want 0", n)
	}
}

// TestStaticLowerCaseFallback: a capitalized query token matches a lower-case
// model entry via the lower-case fallback.
func TestStaticLowerCaseFallback(t *testing.T) {
	s := loadTiny(t)
	if _, ok := s.lookup("CAT"); !ok {
		t.Fatal("CAT should resolve to cat via lower-case fallback")
	}
}

func TestStaticParseErrors(t *testing.T) {
	cases := []struct{ name, in string }{
		{"empty", ""},
		{"bad header", "notanumber\ncat 1 2 3\n"},
		{"short line", "1 3\ncat 1.0 2.0\n"},
		{"bad float", "1 3\ncat 1.0 x 3.0\n"},
		{"dup word", "2 2\ncat 1 0\ncat 0 1\n"},
		{"zero dim", "1 0\n"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := LoadStatic(strings.NewReader(c.in)); err == nil {
				t.Fatalf("expected error for %q", c.name)
			}
		})
	}
}

// TestStaticMultiWordEntry: a word containing a space (last dim fields are the
// vector) parses with the space preserved.
func TestStaticMultiWordEntry(t *testing.T) {
	s, err := LoadStatic(strings.NewReader("1 2\nNew York 3.0 4.0\n"))
	if err != nil {
		t.Fatal(err)
	}
	if v, ok := s.Vector("New York"); !ok || v[0] != 3 || v[1] != 4 {
		t.Fatalf("Vector(New York) = %v, %v", v, ok)
	}
}

// TestStaticSemanticsBeatHashingOnSynonyms is the honest contrast on a TOY
// model: two synonyms with no shared characters get real similarity from Static
// (their vectors are close by construction) but ZERO from Hashing (no shared
// n-grams). This is the semantic vs lexical distinction, shown concretely.
func TestStaticSemanticsBeatHashingOnSynonyms(t *testing.T) {
	// "รถ" and "ยานพาหนะ" placed at nearly the same point; "แมว" far away.
	model := "3 2\nรถ 1.0 0.1\nยานพาหนะ 0.9 0.15\nแมว -1.0 0.0\n"
	s, err := LoadStatic(strings.NewReader(model))
	if err != nil {
		t.Fatal(err)
	}
	synStatic := cosine(s.Embed("รถ"), s.Embed("ยานพาหนะ"))
	if synStatic < 0.9 {
		t.Fatalf("static synonym cos = %.3f, want high", synStatic)
	}
	h := NewHashing(1024)
	synHash := cosine(h.Embed("รถ"), h.Embed("ยานพาหนะ"))
	// No shared characters ⇒ any similarity is pure hash collision, and it must
	// be far below the real semantic similarity Static gives.
	if synHash > 0.1 || synHash > synStatic-0.5 {
		t.Fatalf("hashing gave synonyms cos = %.3f (static = %.3f); expected ~0 (no shared chars)", synHash, synStatic)
	}
}
