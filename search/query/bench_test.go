package query

import "testing"

// benchQueries mixes the two problem extremes: one long conversational query
// (filler + buried entities) and several short bare-acronym queries.
var benchQueries = []string{
	"ช่วยสรุป SOP ที่เกี่ยวการจัดซื้อจัดจ้างหน่อย เอาแผนก it กับ pc นะ",
	"ขอเบอร์ปลาit หน่อย",
	"กุ้งpos อยู่แผนกไหน",
	"it", "pc", "hr", "mkt",
}

func BenchmarkParse(b *testing.B) {
	p := New()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = p.Parse(benchQueries[i%len(benchQueries)])
	}
}

func BenchmarkParseExpand(b *testing.B) {
	p := New()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = p.ParseExpand(benchQueries[i%len(benchQueries)])
	}
}

func BenchmarkExpand(b *testing.B) {
	p := New()
	terms := []string{"IT", "pc", "HR", "sop", "การตลาด", "xyz"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = p.Expand(terms[i%len(terms)])
	}
}
