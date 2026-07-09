//go:build amd64

#include "textflag.h"

// func dotF32U8AVX2(q *float32, b *byte, n int) float32
//
// Sums q[i]*float32(b[i]) with b treated as UNSIGNED bytes, eight per iteration:
// zero-extend 8 bytes to int32 (VPMOVZXBD), convert to float32 (VCVTDQ2PS),
// multiply by q, accumulate. Precondition: n is a positive multiple of 8. The
// eight lanes are horizontally summed at the end (a few ULP vs a sequential sum).
TEXT ·dotF32U8AVX2(SB), NOSPLIT, $0-28
	MOVQ q+0(FP), AX
	MOVQ b+8(FP), BX
	MOVQ n+16(FP), CX
	SHRQ $3, CX          // CX = n / 8
	VXORPS Y0, Y0, Y0    // accumulator

loop:
	VPMOVZXBD (BX), Y1   // 8 uint8 -> 8 uint32
	VCVTDQ2PS Y1, Y1     // -> 8 float32 (0..255)
	VMULPS  (AX), Y1, Y1 // q * float(byte)
	VADDPS  Y1, Y0, Y0   // accumulate
	ADDQ $8, BX          // 8 bytes
	ADDQ $32, AX         // 8 float32
	DECQ CX
	JNZ  loop

	VEXTRACTF128 $1, Y0, X1
	VADDPS  X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	VMOVSS  X0, ret+24(FP)
	VZEROUPPER
	RET

// func dotF32U8FMA(q *float32, b *byte, n int) float32
//
// FMA (VFMADD231PS) variant of dotF32U8AVX2 with four accumulators (32/iter) to
// hide FMA latency. Precondition: n is a positive multiple of 8.
TEXT ·dotF32U8FMA(SB), NOSPLIT, $0-28
	MOVQ q+0(FP), AX
	MOVQ b+8(FP), BX
	MOVQ n+16(FP), CX
	VXORPS Y0, Y0, Y0
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4

	MOVQ CX, DX
	SHRQ $5, DX
	JZ   u8ftail

u8fmain:
	VPMOVZXBD (BX), Y1
	VCVTDQ2PS Y1, Y1
	VFMADD231PS (AX), Y1, Y0
	VPMOVZXBD 8(BX), Y5
	VCVTDQ2PS Y5, Y5
	VFMADD231PS 32(AX), Y5, Y2
	VPMOVZXBD 16(BX), Y1
	VCVTDQ2PS Y1, Y1
	VFMADD231PS 64(AX), Y1, Y3
	VPMOVZXBD 24(BX), Y5
	VCVTDQ2PS Y5, Y5
	VFMADD231PS 96(AX), Y5, Y4
	ADDQ $32, BX
	ADDQ $128, AX
	DECQ DX
	JNZ  u8fmain

	VADDPS Y2, Y0, Y0
	VADDPS Y4, Y3, Y3
	VADDPS Y3, Y0, Y0

u8ftail:
	ANDQ $31, CX
	SHRQ $3, CX
	JZ   u8freduce

u8ftailloop:
	VPMOVZXBD (BX), Y1
	VCVTDQ2PS Y1, Y1
	VFMADD231PS (AX), Y1, Y0
	ADDQ $8, BX
	ADDQ $32, AX
	DECQ CX
	JNZ  u8ftailloop

u8freduce:
	VEXTRACTF128 $1, Y0, X1
	VADDPS  X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	VMOVSS  X0, ret+24(FP)
	VZEROUPPER
	RET
