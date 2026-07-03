package cjk

import "testing"

func BenchmarkCut(b *testing.B) {
	const s = "小明硕士毕业于中国科学院计算所后来到网易杭研大厦从事自然语言处理和机器学习研究"
	Cut("预热") // warm shared load out of timed loop
	b.ReportAllocs()
	b.ResetTimer()
	var n int
	for i := 0; i < b.N; i++ {
		n += len(Cut(s))
	}
	_ = n
}
