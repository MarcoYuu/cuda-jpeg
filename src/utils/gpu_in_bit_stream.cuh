/******************************************************
 CInBitStream Ver.1.0.0 ビットストリーム・クラス
 Copyright 2004 AsWe.Co.,Ltd. All rights reserved.
 # The comments are written in Japanese (Shift-JIS).
 ******************************************************/

#ifndef GPU_IN_BIT_STREAM_H_
#define GPU_IN_BIT_STREAM_H_

#include <stdio.h>
#include <cuda_runtime.h>

#include "../cuda_jpeg_types.h"


#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __host__
#endif

//Max_Block_size:ハフマンエンコードの時のマクロブロック毎のバッファのサイズ,コンスタントのがいいかも
#define MBS 128

//Max_Block_size:ハフマンエンコードの時のマクロブロック毎のバッファのサイズ,コンスタントのがいいかも
//#define MBS 256

class GPUInBitStream {
public:
	int mBytePos;
	int mBitPos;
	int mNumBits;
	int mWriteFlag;

	// コンストラクタ
	GPUInBitStream();
};

//デバイスのアドレスを指すポインタはホストに定義しなければならないっぽい
struct GPUCInBitStream_BufP {
	byte* mHeadOfBufP; // バッファの先頭アドレス
	byte* mBufP; // 書き込みアドレス
};

__device__ void IncBuf(GPUInBitStream *d, byte *mBufP);

// ビット単位で読み出す
// 読み出したビット値：エラーコード
// 読み出すビット数
__device__ int gpu_GetBits(GPUInBitStream *d, byte *mBufP, int numOfBits);


#endif