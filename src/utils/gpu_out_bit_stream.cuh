/******************************************************
 COutBitStream Ver.1.1.0 出力ビットストリーム・クラス
 Copyright 2004 AsWe.Co.Ltd. All right reseved.
 # The comments are written in Japanese (Shift-JIS).
 ******************************************************/

#ifndef GPU_OUT_BIT_STREAM_H_
#define GPU_OUT_BIT_STREAM_H_

#include "../type_definitions.h"
#include "cuda_memory.hpp"

#include <stdio.h>

#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __host__
#endif

//------------------------------------------------------------
//Max_Block_size:ハフマンエンコードの時のマクロブロック毎のバッファのサイズ,コンスタントのがいいかも
//------------------------------------------------------------
#define MBS 128
//#define MBS 256

class GPUOutBitStream {
public:
	int mBytePos;
	int mBitPos; // 書き込みビット位置（上位ビットが7、下位ビットが0）
	int mWriteFlag; // 1:書き込み可, 0:書き込み不可
	int mNumBits; //全体バッファに書き込むサイズ

	GPUOutBitStream();
};

//デバイスのアドレスを指すポインタはホストに定義しなければならないっぽい
class GPUOutBitStreamBufferPointer {
private:
//	byte* _head_of_buff; // バッファの先頭アドレス
//	byte* _write_buff_address; // 書き込みアドレス
//	byte* _end_of_buff; // バッファの終了アドレス

	DeviceMemory<byte> _head_of_buff; // バッファの先頭アドレス
	byte* _write_buff_address; // 書き込みアドレス
	byte* _end_of_buff; // バッファの終了アドレス

	size_t _buff_size;

public:
	GPUOutBitStreamBufferPointer(size_t buff_size) :
		_buff_size(buff_size),
		_head_of_buff(buff_size) {

		_head_of_buff.fillZero();
		_write_buff_address = _head_of_buff.device_data();
		_end_of_buff = _head_of_buff.device_data() + _buff_size; // バッファの最終アドレス
	}

	byte* getHeadOfBuffer() {
		return _head_of_buff.device_data();
	}

	byte* getEndOfBuf() {
		return _end_of_buff;
	}

	byte* getWriteBufAddress() {
		return _write_buff_address;
	}

	const byte* getHeadOfBuffer() const {
		return _head_of_buff.device_data();
	}

	const byte* getEndOfBuf() const {
		return _end_of_buff;
	}

	const byte* getWriteBufAddress() const {
		return _write_buff_address;
	}
};

//void InitGPUBuffer(GPUOutBitStreamBufferPointer* bp, int size);

__device__ void IncBuf(GPUOutBitStream *d, byte *mBufP, byte *mEndOfBufP);

//------------------------------------------------------------
// 8ビット以下のデータを１つのアドレスに書き込む
//------------------------------------------------------------
__device__
void SetFewBits(GPUOutBitStream *d, byte *mBufP, byte *mEndOfBufP,
	byte v, int numBits);

//------------------------------------------------------------
// 8ビット以下のデータを2つのアドレスに分けて書き込む
//------------------------------------------------------------
__device__
void SetBits2Byte(GPUOutBitStream *d, byte *mBufP, byte *mEndOfBufP,
	byte v, int numBits);

//------------------------------------------------------------
// 8ビット以下のデータを書き込む
//------------------------------------------------------------
__device__
void Set8Bits(GPUOutBitStream *d, byte *mBufP, byte *mEndOfBufP,
	byte v, int numBits);

//------------------------------------------------------------
// ビット単位で書き込む
//
//		vには右詰めでデータが入っている。
//		mBufPには左詰めで格納する
//		vが負の時下位ビットのみを格納する
//		numBitsは1以上16以下
//
//		v	00010011(2)のとき
//		10011を*mBufPに追加する
//
// ビット単位で書き込む（最大16ビット）
//------------------------------------------------------------
__device__
void SetBits(GPUOutBitStream *d, byte *mBufP, byte *mEndOfBufP,
	int v, int numBits);

////////////////////////////////////////////////////////////////////
//MCU毎から1枚のバッファへ。上位ビットから書き込むので上の流用不可//
////////////////////////////////////////////////////////////////////

__device__
void IncBuf_w(GPUOutBitStream *d);

//------------------------------------------------------------
// 8ビット以下のデータを１つのアドレスに書き込む
//------------------------------------------------------------
__device__
void SetFewBits_w(GPUOutBitStream *d, byte *mBufP, byte v,
	int numBits);

//------------------------------------------------------------
// 8ビット以下のデータを2つのアドレスに分けて書き込む
//------------------------------------------------------------
__device__
void SetBits2Byte_w(GPUOutBitStream *d, byte *mBufP, byte v,
	int numBits);
__device__
void Set8Bits_w(GPUOutBitStream *d, byte *mBufP, byte v,
	int numBits);

//------------------------------------------------------------
// GPUCOutBitStream *Od, //マクロブロック毎の書き込み位置と書き込みbit数を記録してある
// byte *OmBufP,//書き込み先先頭アドレス、一枚のバッファ
// byte *ImBufP,// 書き込み元先頭アドレス、マクロ毎のバッファ
// int id//バグチェック用に使ってた
//------------------------------------------------------------
__device__
void WriteBits(GPUOutBitStream *Od, byte *OmBufP, byte *ImBufP,
	int id);

//------------------------------------------------------------
// 余ったビットに1を詰めるためのマスク,COutBitStream.hで定義してる
//------------------------------------------------------------
__device__
static const byte GPUkBitFullMaskT[8] = { 0x01, 0x03, 0x07, 0x0f,
	0x1f, 0x3f, 0x7f, 0xff };
__device__
static const byte GPUkBitFullMaskLowT[8] = { 0x80, 0xc0, 0xe0, 0xf0,
	0xf8, 0xfc, 0xfe, 0xff };

inline __device__ void IncBuf(GPUOutBitStream *d, byte *mBufP,
	byte *mEndOfBufP) {
	//エラー検出、ちょっとだけ遅くなる
	if (++d->mBytePos >= MBS) {
		printf("IncBuf:buff_overflow");
	}
}

// 8ビット以下のデータを１つのアドレスに書き込む
inline __device__ void SetFewBits(GPUOutBitStream *d, byte *mBufP,
	byte *mEndOfBufP, byte v, // 書き込む値
	int numBits) // 書き込みビット数
	{
	// 上位ビットをクリア
	v &= GPUkBitFullMaskT[numBits - 1];
	*(mBufP + d->mBytePos) |= v << (d->mBitPos + 1 - numBits);
	if ((d->mBitPos -= numBits) < 0) {
		IncBuf(d, mBufP, mEndOfBufP);
		d->mBitPos = 7;
	}
}

// 8ビット以下のデータを2つのアドレスに分けて書き込む
inline __device__ void SetBits2Byte(GPUOutBitStream *d, byte *mBufP,
	byte *mEndOfBufP, byte v, // 書き込む値
	int numBits) // 書き込みビット数
	{
	v &= GPUkBitFullMaskT[numBits - 1]; // 上位ビットをクリア
	int nextBits = numBits - (d->mBitPos + 1); // 次のバイトに入れるビット数
	*(mBufP + d->mBytePos) |= (v >> nextBits) & GPUkBitFullMaskT[d->mBitPos]; // 1バイト目書き込み

	IncBuf(d, mBufP, mEndOfBufP);
	*(mBufP + d->mBytePos) |= v << (8 - nextBits); // 2バイト目書き込み
	d->mBitPos = 7 - nextBits; // ビット位置更新
}

// 8ビット以下のデータを書き込む
inline __device__ void Set8Bits(GPUOutBitStream *d, byte *mBufP,
	byte *mEndOfBufP, byte v, // 書き込む値
	int numBits) // 書き込みビット数
	{
	if (d->mBitPos + 1 >= numBits) // 現在のバイトに全部入るとき
		SetFewBits(d, mBufP, mEndOfBufP, (byte) v, numBits);
	else
		// 現在のバイトからはみ出すとき
		SetBits2Byte(d, mBufP, mEndOfBufP, (byte) v, numBits);
}

// ビット単位で書き込む
//
//		vには右詰めでデータが入っている。
//		mBufPには左詰めで格納する
//		vが負の時下位ビットのみを格納する
//		numBitsは1以上16以下
//
//		v	00010011(2)のとき
//		10011を*mBufPに追加する
// ビット単位で書き込む（最大16ビット）
inline __device__ void SetBits(GPUOutBitStream *d, byte *mBufP,
	byte *mEndOfBufP, int v, // 書き込む値
	int numBits) // 書き込みビット数
	{
	if (numBits == 0)
		return; // 何もしない
	//if( numBits > 16 )
	//	throw( EBufferError( "SetBits:Max Bit Over" ) );

	if (numBits > 8) { // 2バイトなら
		Set8Bits(d, mBufP, mEndOfBufP, byte(v >> 8), numBits - 8); // 上位バイト書き込み
		numBits = 8;
	}
	Set8Bits(d, mBufP, mEndOfBufP, byte(v), numBits); // 残り下位バイトを書き込み
	//d->mWriteNum++;//バッファを使った回数を追加
}

////////////////////////////////////////////////////////////////////
//MCU毎から1枚のバッファへ。上位ビットから書き込むので上の流用不可//
////////////////////////////////////////////////////////////////////
inline __device__ void IncBuf_w(GPUOutBitStream *d) {
	++d->mBytePos;
}

// 8ビット以下のデータを１つのアドレスに書き込む
inline __device__ void SetFewBits_w(GPUOutBitStream *d, byte *mBufP, byte v, // 書き込む値
	int numBits) // 書き込みビット数
	{
	v &= GPUkBitFullMaskLowT[numBits - 1]; // 下位ビットをクリア
	*(mBufP + d->mBytePos) |= (v >> 7 - d->mBitPos); //

	if ((d->mBitPos -= numBits) < 0) {
		//マーカが存在しないのでいらない、デコーダも修正

		IncBuf_w(d);
		d->mBitPos = 7;
	}
}

// 8ビット以下のデータを2つのアドレスに分けて書き込む
inline __device__ void SetBits2Byte_w(GPUOutBitStream *d, byte *mBufP, byte v, // 書き込む値
	int numBits) // 書き込みビット数
	{
	v &= GPUkBitFullMaskLowT[numBits - 1]; // 下位ビットをクリア
	int nextBits = numBits - (d->mBitPos + 1); // 次のバイトに入れるビット数

	*(mBufP + d->mBytePos) |= (v >> 7 - d->mBitPos); // 1バイト目書き込み

	IncBuf_w(d);
	*(mBufP + d->mBytePos) |= (v << d->mBitPos + 1); // 2バイト目書き込み
	d->mBitPos = 7 - nextBits; // ビット位置更新
}

inline __device__ void Set8Bits_w(GPUOutBitStream *d, byte *mBufP, byte v, // 書き込む値
	int numBits) // 書き込みビット数
	{
	if (d->mBitPos + 1 >= numBits) // 現在のバイトに全部入るとき
		SetFewBits_w(d, mBufP, (byte) v, numBits);
	else
		// 現在のバイトからはみ出すとき
		SetBits2Byte_w(d, mBufP, (byte) v, numBits);
}

inline __device__ void WriteBits(GPUOutBitStream *Od, //マクロブロック毎の書き込み位置と書き込みbit数を記録してある
	byte *OmBufP, //書き込み先先頭アドレス、一枚のバッファ
	byte *ImBufP, // 書き込み元先頭アドレス、マクロ毎のバッファ
	int id //バグチェック用に使ってた
	) {
	int bytepos = 0;
	while (Od->mNumBits > 8) {
		Set8Bits_w(Od, OmBufP, *(ImBufP + bytepos), 8); // 1バイト書き込み
		Od->mNumBits -= 8;
		bytepos++;
	}

	Set8Bits_w(Od, OmBufP, *(ImBufP + bytepos), Od->mNumBits); // 端数バイト書き込み
}

#endif
