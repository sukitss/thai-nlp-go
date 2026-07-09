//go:build amd64

#include "textflag.h"

// func dotFloat32AVX2(a, b *float32, n int) float32
//
// Sums the pairwise products of a[0:n] and b[0:n] using AVX2, eight float32 per
// iteration. Precondition (guaranteed by the Go caller): n is a positive
// multiple of 8. The eight lane accumulators are horizontally summed at the end,
// so the result can differ from a strict left-to-right sum by a few ULP.
TEXT ·dotFloat32AVX2(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), AX
	MOVQ b+8(FP), BX
	MOVQ n+16(FP), CX
	SHRQ $3, CX          // CX = n / 8 (iteration count)
	VXORPS Y0, Y0, Y0    // accumulator: 8 x float32 = 0

loop:
	VMOVUPS (AX), Y1     // 8 float32 from a
	VMULPS  (BX), Y1, Y1 // Y1 = a * b
	VADDPS  Y1, Y0, Y0   // accumulate
	ADDQ $32, AX
	ADDQ $32, BX
	DECQ CX
	JNZ  loop

	// Horizontal sum of the 8 float32 lanes in Y0 -> a single float32.
	VEXTRACTF128 $1, Y0, X1
	VADDPS  X1, X0, X0   // 4 float32
	VHADDPS X0, X0, X0   // 2 sums
	VHADDPS X0, X0, X0   // 1 sum
	VMOVSS  X0, ret+24(FP)
	VZEROUPPER
	RET

// func dotFloat32FMA(a, b *float32, n int) float32
//
// Fuses multiply and add into VFMADD231PS. FMA has higher latency than a bare
// VADDPS, so a single accumulator would be latency-bound and slower than the
// AVX2 kernel; four independent accumulators (32 floats/iteration) hide that
// latency. Precondition: n is a positive multiple of 8.
TEXT ·dotFloat32FMA(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), AX
	MOVQ b+8(FP), BX
	MOVQ n+16(FP), CX
	VXORPS Y0, Y0, Y0
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4

	MOVQ CX, DX
	SHRQ $5, DX          // DX = n / 32 (4-accumulator main loop)
	JZ   fmatail

fmamain:
	VMOVUPS (AX), Y1
	VFMADD231PS (BX), Y1, Y0
	VMOVUPS 32(AX), Y5
	VFMADD231PS 32(BX), Y5, Y2
	VMOVUPS 64(AX), Y1
	VFMADD231PS 64(BX), Y1, Y3
	VMOVUPS 96(AX), Y5
	VFMADD231PS 96(BX), Y5, Y4
	ADDQ $128, AX
	ADDQ $128, BX
	DECQ DX
	JNZ  fmamain

	VADDPS Y2, Y0, Y0
	VADDPS Y4, Y3, Y3
	VADDPS Y3, Y0, Y0

fmatail:
	ANDQ $31, CX         // CX = n % 32
	SHRQ $3, CX          // remaining 8-blocks
	JZ   fmareduce

fmatailloop:
	VMOVUPS (AX), Y1
	VFMADD231PS (BX), Y1, Y0
	ADDQ $32, AX
	ADDQ $32, BX
	DECQ CX
	JNZ  fmatailloop

fmareduce:
	VEXTRACTF128 $1, Y0, X1
	VADDPS  X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	VMOVSS  X0, ret+24(FP)
	VZEROUPPER
	RET
