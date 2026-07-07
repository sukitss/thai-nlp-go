package embed

import (
	"fmt"
	"strings"
	"testing"
)

// benchDoc is a representative mixed Thai/English/acronym passage.
const benchDoc = "ติดต่อทีมเน็ตเวิร์ก ปลา IT, พี่ปลา, เบอร์ปลา ติดต่อพี่ปลา (IT) " +
	"โทร 081-234-5678 หรือแจ้งผ่านระบบ ticket ขอสิทธิ์เข้าระบบ POS support"

// BenchmarkHashingEmbed measures embed throughput and allocations at a few
// dimensions (the alloc count is dominated by the token slice and the n-gram
// string builder, independent of dim).
func BenchmarkHashingEmbed(b *testing.B) {
	for _, dim := range []int{128, 256, 512} {
		h := NewHashing(dim)
		b.Run(fmt.Sprintf("dim%d", dim), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = h.Embed(benchDoc)
			}
		})
	}
}

// BenchmarkStaticEmbed measures mean-pool embed throughput over a small in-vocab
// model (lookups + accumulation, no hashing).
func BenchmarkStaticEmbed(b *testing.B) {
	// Build a synthetic 200-word, 100-dim model covering the doc's tokens.
	var sb strings.Builder
	words := []string{"ติดต่อ", "ทีม", "เน็ตเวิร์ก", "ปลา", "it", "พี่", "เบอร์",
		"โทร", "ระบบ", "ticket", "ขอ", "สิทธิ์", "เข้า", "pos", "support"}
	const dim = 100
	fmt.Fprintf(&sb, "%d %d\n", len(words), dim)
	for wi, w := range words {
		sb.WriteString(w)
		for d := 0; d < dim; d++ {
			fmt.Fprintf(&sb, " %.3f", float32((wi*7+d)%13)/13)
		}
		sb.WriteByte('\n')
	}
	s, err := LoadStatic(strings.NewReader(sb.String()))
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = s.Embed(benchDoc)
	}
}
