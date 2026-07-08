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
