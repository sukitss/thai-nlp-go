package keyword

import "testing"

// benchDoc is a representative mixed Thai/English/acronym FAQ paragraph.
const benchDoc = "ติดต่อทีมเน็ตเวิร์ก ปลา IT, พี่ปลา, เบอร์ปลา, ติดต่อปลา " +
	"ติดต่อพี่ปลา (IT) โทร 081-234-5678 หรือแจ้งผ่านระบบ ticket " +
	"ขอสิทธิ์เข้าระบบ POS ขอสิทธิ์ POS, permission, กุ้ง POS " +
	"ติดต่อพี่กุ้ง (POS) โทร 081-999-0000 พร้อมแนบใบอนุมัติ support"

// BenchmarkExtract measures extract throughput (top-10) per algorithm. TF-IDF
// includes idf lookups against a small corpus vocabulary; the others are
// corpus-free.
func BenchmarkExtract(b *testing.B) {
	v := BuildVocab(faqRows)
	engines := []struct {
		name string
		ex   Extractor
	}{
		{"tfidf", NewTFIDFTopK(v)},
		{"rake", NewRAKE()},
		{"yake", NewYAKE()},
		{"textrank", NewTextRank()},
	}
	for _, e := range engines {
		b.Run(e.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = e.ex.Extract(benchDoc, 10)
			}
		})
	}
}
