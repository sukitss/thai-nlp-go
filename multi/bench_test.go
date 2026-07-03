package multi

import "testing"

func BenchmarkSegment(b *testing.B) {
	const s = "ผมอ่านนิยายแปลจีน三国志และมังงะ日本語ที่แปลเป็นไทยกับ한국webtoon"
	Segment("warm")
	b.ReportAllocs()
	b.ResetTimer()
	var n int
	for i := 0; i < b.N; i++ {
		n += len(Segment(s))
	}
	_ = n
}

func BenchmarkAppendBytes(b *testing.B) {
	const s = "ผมอ่านนิยายแปลจีน三国志และมังงะ日本語ที่แปลเป็นไทยกับ한국webtoon"
	buf := make([]byte, 0, 256)
	AppendBytes(buf, "warm", ' ')
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf = AppendBytes(buf[:0], s, ' ')
	}
	_ = buf
}
