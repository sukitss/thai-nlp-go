package chunk

import (
	"math/rand/v2"
	"testing"
)

// FuzzSplit guards Split against panics on arbitrary input and re-asserts
// the core invariants (offset fidelity, MaxUnits, rune/combining-mark
// boundaries, whitespace coverage, overlap bound) under the default Measure
// and built-in splitter. Run with: go test ./chunk/ -fuzz=FuzzSplit
func FuzzSplit(f *testing.F) {
	f.Add("", 1, 0)
	f.Add("สวัสดีครับ วันนี้อากาศดีมาก\nย่อหน้าใหม่ ตามมา", 10, 3)
	f.Add("ก"+string(make([]byte, 3))+"ั้ํ๎", 2, 1)
	f.Add("mixed ไทย eng\r\nกั"+"\xff\xfe"+" ท้าย ", 5, 4)
	r := rand.New(rand.NewPCG(7, 7))
	for range 4 {
		f.Add(genDoc(r), 1+r.IntN(64), r.IntN(8))
	}
	f.Fuzz(func(t *testing.T, text string, maxUnits, overlap int) {
		const capUnits = 64
		if maxUnits < 0 {
			maxUnits = -maxUnits
		}
		maxUnits = maxUnits%capUnits + 1
		if overlap < 0 {
			overlap = -overlap
		}
		overlap %= maxUnits
		o := Options{MaxUnits: maxUnits, OverlapUnits: overlap}
		cs, err := Split(text, o)
		if err != nil {
			t.Fatalf("valid options rejected: %v", err)
		}
		checkInvariants(t, text, cs, o)
	})
}
