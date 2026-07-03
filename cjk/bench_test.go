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

func BenchmarkAppendBytes(b *testing.B) {
	const s = "小明硕士毕业于中国科学院计算所后来到网易杭研大厦从事自然语言处理和机器学习研究"
	seg, _ := Default()
	buf := make([]byte, 0, 256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf = seg.AppendBytes(buf[:0], s, ' ')
	}
	_ = buf
}

func BenchmarkCutParallel(b *testing.B) {
	const s = "小明硕士毕业于中国科学院计算所后来到网易杭研大厦从事自然语言处理和机器学习研究"
	seg, _ := Default() // shared read-only dict across goroutines
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		buf := make([]byte, 0, 256)
		for pb.Next() {
			buf = seg.AppendBytes(buf[:0], s, ' ')
		}
	})
}

func BenchmarkCutDP(b *testing.B) {
	const s = "小明硕士毕业于中国科学院计算所后来到网易杭研大厦从事自然语言处理和机器学习研究"
	seg, _ := Default()
	b.ReportAllocs()
	b.ResetTimer()
	var n int
	for i := 0; i < b.N; i++ {
		n += len(seg.CutDP(s))
	}
	_ = n
}
