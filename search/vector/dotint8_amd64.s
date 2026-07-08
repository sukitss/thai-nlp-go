//go:build amd64

#include "textflag.h"

// func dotInt8AVX2(a, b *byte, n int) int32
//
// Sums the pairwise signed-int8 products of a[0:n] and b[0:n] using AVX2.
// Precondition (guaranteed by the Go caller): n is a positive multiple of 16.
//
// Per iteration it loads 16 signed bytes from each input, sign-extends them to
// int16 (VPMOVSXBW), multiplies and pairwise-adds them into eight int32 lanes
// (VPMADDWD), and accumulates. The eight lanes are horizontally summed at the
// end. int8*int8 <= 16129 and two per VPMADDWD lane <= 32258, so the int32
// accumulator matches the portable loop for any realistic dimensionality.
TEXT ·dotInt8AVX2(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), AX
	MOVQ b+8(FP), BX
	MOVQ n+16(FP), CX
	SHRQ $4, CX          // CX = n / 16 (iteration count)
	VPXOR Y0, Y0, Y0     // accumulator: 8 x int32 = 0

loop:
	VPMOVSXBW (AX), Y1   // 16 int8 from a -> 16 int16
	VPMOVSXBW (BX), Y2   // 16 int8 from b -> 16 int16
	VPMADDWD  Y2, Y1, Y1 // 8 x int32: adjacent (a*b) pairs summed
	VPADDD    Y1, Y0, Y0 // accumulate
	ADDQ $16, AX
	ADDQ $16, BX
	DECQ CX
	JNZ  loop

	// Horizontal sum of the 8 int32 lanes in Y0 -> a single int32.
	VEXTRACTI128 $1, Y0, X1
	VPADDD  X1, X0, X0   // 4 int32
	VPSHUFD $0xEE, X0, X1
	VPADDD  X1, X0, X0   // 2 int32
	VPSHUFD $0x55, X0, X1
	VPADDD  X1, X0, X0   // 1 int32
	VMOVD   X0, AX
	MOVL    AX, ret+24(FP)
	VZEROUPPER
	RET
