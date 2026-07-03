package jp

import "testing"

func BenchmarkCut(b *testing.B) {
	const s = "私は東京大学で自然言語処理と機械学習の研究をしています"
	Cut("予熱")
	b.ReportAllocs()
	b.ResetTimer()
	var n int
	for i := 0; i < b.N; i++ {
		n += len(Cut(s))
	}
	_ = n
}
