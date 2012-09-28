
#include "gpu_in_bit_stream.cuh"

// ビット取り出しのためのマスク
__device__ static const int GPUkBitTestMaskT[8] = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 };

GPUInBitStream::GPUInBitStream() :
mBitPos(7), // 読み出しビット位置（上位ビットが7、下位ビットが0）
	mBytePos(0), // 次のバイトを読んでいいかどうか
	mNumBits(0), // 1:読み出し可, 0:読み出し不可
	mWriteFlag(0) {
}

__device__ void IncBuf(	GPUInBitStream *d, byte *mBufP) {
	//エラー検出、ちょっとだけ遅くなる
	if (++d->mBytePos >= MBS) {
		printf("IncBuf:buff_overflow");
	}
}

// ビット単位で読み出す
// 読み出したビット値：エラーコード
// 読み出すビット数
__device__ int gpu_GetBits(GPUInBitStream *d, byte *mBufP, int numOfBits) {
	if (numOfBits <= 0)
		return 0; // エラー

	int r = 0; // 返値
	while (numOfBits) {

		if (d->mBitPos < 0) { // 次のバイトを読み出すとき
			d->mBitPos = 7; // 読み出しビット位置更新
			IncBuf(d, mBufP); // アドレス更新
		}

		// 返値の作成
		// 1ビット読み出しと読み出しビット位置更新
		r <<= 1;
		r |= (*(mBufP + d->mBytePos) & GPUkBitTestMaskT[d->mBitPos--]) ? 1 : 0;
		// 読み出しビット数更新
		numOfBits--;
	}
	return r;
}
