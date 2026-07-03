package vocab

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"strings"
	"testing"
)

// equalState verifies v has exactly the same terms/ids/df/docs as b.
func equalState(t *testing.T, b *Builder, v *Vocab) {
	t.Helper()
	if v.Len() != b.Len() || v.DocCount() != b.DocCount() {
		t.Fatalf("Len/DocCount = %d/%d, want %d/%d", v.Len(), v.DocCount(), b.Len(), b.DocCount())
	}
	b.Terms(func(term string, id uint32, df int) bool {
		gotID, ok := v.ID(term)
		if !ok || gotID != id {
			t.Errorf("ID(%q) = %d,%v, want %d,true", term, gotID, ok, id)
		}
		if got := v.DF(term); got != df {
			t.Errorf("DF(%q) = %d, want %d", term, got, df)
		}
		return true
	})
}

func TestRoundTripEmpty(t *testing.T) {
	b := NewBuilder()
	v := saveLoad(t, b)
	equalState(t, b, v)
	if _, ok := v.ID("อะไรก็ตาม"); ok {
		t.Error("empty vocab should contain nothing")
	}
}

func TestRoundTripOneTerm(t *testing.T) {
	b := NewBuilder()
	b.AddDoc([]string{"เดียว"})
	v := saveLoad(t, b)
	equalState(t, b, v)
}

// TestRoundTrip10k: 10k terms including Thai, emoji, and the empty string ""
// (a legal, distinct term — the package does not police tokenization).
func TestRoundTrip10k(t *testing.T) {
	b := NewBuilder()
	doc := make([]string, 0, 100)
	for i := 0; i < 10000; i++ {
		var term string
		switch i % 4 {
		case 0:
			term = fmt.Sprintf("คำที่%d", i)
		case 1:
			term = fmt.Sprintf("word%d", i)
		case 2:
			term = fmt.Sprintf("🐱%d้", i) // emoji + combining Thai tone mark
		default:
			term = fmt.Sprintf("ผสมmix%d", i)
		}
		doc = append(doc, term)
		if len(doc) == 100 {
			b.AddDoc(doc)
			doc = doc[:0]
		}
	}
	b.AddDoc([]string{"", "ท้าย"}) // empty-string term
	if b.Len() != 10002 {
		t.Fatalf("Len = %d, want 10002", b.Len())
	}
	v := saveLoad(t, b)
	equalState(t, b, v)
	if df := v.DF(""); df != 1 {
		t.Errorf(`DF("") = %d, want 1`, df)
	}
}

// TestSaveDeterministic: Save of the same state is byte-identical, and so is
// Save after a Load/Extend cycle (the format has one canonical encoding).
func TestSaveDeterministic(t *testing.T) {
	b := NewBuilder()
	b.AddDoc([]string{"หนึ่ง", "สอง", "🐱", ""})
	b.AddDoc([]string{"สอง", "สาม"})
	var one, two, three bytes.Buffer
	if err := b.Save(&one); err != nil {
		t.Fatal(err)
	}
	if err := b.Save(&two); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(one.Bytes(), two.Bytes()) {
		t.Error("two Saves of the same builder differ")
	}
	v, err := Load(bytes.NewReader(one.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if err := v.Extend().Save(&three); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(one.Bytes(), three.Bytes()) {
		t.Error("Save → Load → Extend → Save is not byte-identical")
	}
}

// validSnapshot returns Save bytes for a small vocabulary used by the corrupt
// and truncation tests.
func validSnapshot(t testing.TB) []byte {
	t.Helper()
	b := NewBuilder()
	b.AddDoc([]string{"แมว", "วิ่ง", "เร็ว"})
	b.AddDoc([]string{"แมว", "นอน"})
	var buf bytes.Buffer
	if err := b.Save(&buf); err != nil {
		t.Fatalf("Save: %v", err)
	}
	return buf.Bytes()
}

// raw builds an arbitrary (possibly invalid) snapshot byte stream for the
// corrupt-input table: header + uvarint fields + literal term bytes.
func raw(fields ...any) []byte {
	var out []byte
	out = binary.LittleEndian.AppendUint32(out, vocabMagic)
	out = append(out, vocabVersion)
	for _, f := range fields {
		switch x := f.(type) {
		case int:
			out = binary.AppendUvarint(out, uint64(x))
		case uint64:
			out = binary.AppendUvarint(out, x)
		case string:
			out = append(out, x...)
		case []byte:
			out = append(out, x...)
		default:
			panic("raw: bad field type")
		}
	}
	return out
}

func TestLoadCorrupt(t *testing.T) {
	valid := validSnapshot(t)
	overflowVarint := bytes.Repeat([]byte{0xFF}, 10) // > 64 bits
	cases := []struct {
		name    string
		data    []byte
		wantErr string // substring the error must contain
	}{
		{"empty", nil, "truncated header"},
		{"header only 4 bytes", valid[:4], "truncated header"},
		{"bad magic", append([]byte("NOPE"), valid[4:]...), "bad magic"},
		{"future version", append(append([]byte{}, valid[:4]...), append([]byte{99}, valid[5:]...)...), "unsupported format version"},
		{"missing doc count", raw(), "doc count"},
		{"doc count overflows uvarint", raw(overflowVarint), "doc count"},
		{"doc count overflows int", raw(uint64(1) << 63), "overflows int"},
		{"missing term count", raw(0), "term count"},
		{"term count exceeds file", raw(0, 1000000), "exceeds file size"},
		{"term count overflows uvarint", raw(0, overflowVarint), "term count"},
		{"missing term length", raw(0, 1), "term count"}, // 1 term needs ≥2 bytes, 0 remain
		{"term length exceeds file", raw(1, 1, 100, "hi", 1), "exceeds remaining"},
		{"term length overflows uvarint", raw(1, 1, overflowVarint), "term length"},
		{"missing df", raw(1, 1, 2, "hi"), "df"},
		{"df exceeds doc count", raw(1, 1, 2, "hi", 2), "exceeds doc count"},
		{"duplicate term", raw(1, 2, 2, "hi", 1, 2, "hi", 1), "duplicate term"},
		{"trailing garbage", append(append([]byte{}, valid...), 0x00), "trailing"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			v, err := Load(bytes.NewReader(c.data))
			if err == nil {
				t.Fatalf("Load succeeded (Len=%d), want error containing %q", v.Len(), c.wantErr)
			}
			if !strings.Contains(err.Error(), c.wantErr) {
				t.Errorf("error = %q, want substring %q", err, c.wantErr)
			}
		})
	}
	// sanity: the valid snapshot itself loads
	if _, err := Load(bytes.NewReader(valid)); err != nil {
		t.Fatalf("valid snapshot failed to load: %v", err)
	}
}

// TestLoadEveryTruncation: every strict prefix of a valid snapshot must be
// rejected (the format leaves no ambiguity about where the data ends).
func TestLoadEveryTruncation(t *testing.T) {
	valid := validSnapshot(t)
	for i := 0; i < len(valid); i++ {
		if _, err := Load(bytes.NewReader(valid[:i])); err == nil {
			t.Errorf("truncation to %d/%d bytes loaded without error", i, len(valid))
		}
	}
}

// FuzzLoad: Load must never panic, and anything it accepts must survive a
// re-save/re-load round trip unchanged.
func FuzzLoad(f *testing.F) {
	f.Add([]byte(nil))
	f.Add(validSnapshot(f))
	b := NewBuilder()
	b.AddDoc([]string{"", "🐱", "คำ"})
	var buf bytes.Buffer
	if err := b.Save(&buf); err != nil {
		f.Fatal(err)
	}
	f.Add(buf.Bytes())
	f.Add(raw(1, 1, 2, "hi", 2))
	f.Add(bytes.Repeat([]byte{0xFF}, 32))
	f.Fuzz(func(t *testing.T, data []byte) {
		v, err := Load(bytes.NewReader(data))
		if err != nil {
			return
		}
		var out bytes.Buffer
		if err := v.Extend().Save(&out); err != nil {
			t.Fatalf("re-save of loaded vocab failed: %v", err)
		}
		v2, err := Load(&out)
		if err != nil {
			t.Fatalf("re-load of re-saved vocab failed: %v", err)
		}
		if v2.Len() != v.Len() || v2.DocCount() != v.DocCount() {
			t.Fatalf("round trip changed state: %d/%d → %d/%d", v.Len(), v.DocCount(), v2.Len(), v2.DocCount())
		}
		v.Terms(func(term string, id uint32, df int) bool {
			id2, ok := v2.ID(term)
			if !ok || id2 != id || v2.DF(term) != df {
				t.Fatalf("round trip changed term %q", term)
			}
			return true
		})
	})
}
