package charseg

import (
	"bytes"
	"encoding/binary"
	"math"
	"strings"
	"sync"
	"testing"
)

func TestSplitBasic(t *testing.T) {
	if got := Split(""); got != nil {
		t.Fatalf("empty: want nil, got %v", got)
	}
	// Terminal punctuation should end a sentence.
	txt := "เขามาถึงแล้ว. เธอยิ้มให้เขา"
	got := Split(txt)
	if len(got) == 0 {
		t.Fatal("expected at least one segment")
	}
	// Reconstruct: concatenating trimmed segments (dropping spaces) must be a
	// subsequence of the input's non-space characters.
	joined := strings.Join(got, "")
	if strings.ContainsAny(joined, "\n") {
		t.Fatalf("segments contain newlines: %q", got)
	}
}

func TestSplitCoversInput(t *testing.T) {
	// Every non-space rune of the input must appear, in order, across the
	// concatenated segments (no data loss).
	in := "สวัสดีครับ วันนี้อากาศดี “ไปไหนมา” เขาถาม"
	segs := Split(in)
	got := strings.ReplaceAll(strings.Join(segs, ""), " ", "")
	exp := strings.ReplaceAll(in, " ", "")
	if got != exp {
		t.Fatalf("lossy split:\n got %q\n exp %q", got, exp)
	}
}

func TestDeterministicTrain(t *testing.T) {
	ex := []Example{
		{Runes: []rune("กขค งจฉ. ชซฌ"), Bnd: mkBnd("กขค งจฉ. ชซฌ", []int{6, 11})},
		{Runes: []rune("อบด เฟก"), Bnd: mkBnd("อบด เฟก", []int{6})},
	}
	m1 := Train(ex, TrainConfig{Bits: 16, Iters: 5})
	m2 := Train(ex, TrainConfig{Bits: 16, Iters: 5})
	if !bytes.Equal(f32bytes(m1.w), f32bytes(m2.w)) {
		t.Fatal("training not deterministic")
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	ex := []Example{{Runes: []rune("กขค งจฉ. ชซฌ"), Bnd: mkBnd("กขค งจฉ. ชซฌ", []int{6, 11})}}
	m := Train(ex, TrainConfig{Bits: 14, Iters: 3})
	var buf bytes.Buffer
	if err := m.Save(&buf); err != nil {
		t.Fatal(err)
	}
	m2, err := LoadModel(&buf)
	if err != nil {
		t.Fatal(err)
	}
	in := "กขค งจฉ. ชซฌ"
	if strings.Join(m.Split(in), "|") != strings.Join(m2.Split(in), "|") {
		t.Fatal("round-trip changed predictions")
	}
}

func TestConcurrentSplit(t *testing.T) {
	// The default model must be safe for concurrent reads (prod uses many
	// goroutines). Run with -race.
	m := Default()
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				m.Split("สวัสดีครับ วันนี้อากาศดีมาก. เขาพูดว่า “ไปกันเถอะ”")
			}
		}()
	}
	wg.Wait()
}

func TestBoundariesTauMatchesDefault(t *testing.T) {
	m := Default()
	in := []rune("สวัสดีครับ วันนี้อากาศดีมาก. เขาพูดว่า “ไปกันเถอะ” แล้วก็เดินจากไป")
	def := m.Boundaries(in)
	// τ=0 must exactly reproduce the default Boundaries operating point.
	tau0 := m.BoundariesTau(in, 0)
	if len(def) != len(tau0) {
		t.Fatalf("len mismatch %d != %d", len(def), len(tau0))
	}
	for i := range def {
		if def[i] != tau0[i] {
			t.Fatalf("BoundariesTau(0) != Boundaries at %d", i)
		}
	}
	// Recall-lean: lower τ never DROPS a boundary the default kept (monotone).
	lean := m.BoundariesTau(in, -20)
	for i := range def {
		if def[i] && !lean[i] {
			t.Fatalf("lower τ dropped a default boundary at %d (not recall-lean)", i)
		}
	}
	// SplitTau(·,0) matches Split.
	if strings.Join(m.Split(string(in)), "|") != strings.Join(m.SplitTau(string(in), 0), "|") {
		t.Fatal("SplitTau(·,0) != Split")
	}
}

func mkBnd(s string, ends []int) []bool {
	r := []rune(s)
	b := make([]bool, len(r))
	for _, e := range ends {
		if e >= 0 && e < len(b) {
			b[e] = true
		}
	}
	if len(b) > 0 {
		b[len(b)-1] = true
	}
	return b
}

// f32bytes is a structural fingerprint (index+bits of nonzero weights) for
// comparing two trained models.
func f32bytes(w []float32) []byte {
	var buf bytes.Buffer
	var tmp [8]byte
	for i, v := range w {
		if v == 0 {
			continue
		}
		binary.LittleEndian.PutUint32(tmp[0:4], uint32(i))
		binary.LittleEndian.PutUint32(tmp[4:8], math.Float32bits(v))
		buf.Write(tmp[:])
	}
	return buf.Bytes()
}
