/******************************************************
 COutBitStream Ver.1.1.0 ビットストリーム・クラス
 Copyright 2004 AsWe.Co.Ltd. All rights reserved.
 # The comments are written in Japanese (Shift-JIS).
 ******************************************************/

#include <cuda_runtime.h>
#include <cstring>

#include "gpu_out_bit_stream.cuh"


#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __host__
#endif

GPUOutBitStreamState::GPUOutBitStreamState() :
	mBytePos(0),
	mBitPos(7), // 最上位ビット
	mWriteFlag(1), // アクセス許可
	mNumBits(0) {
}

