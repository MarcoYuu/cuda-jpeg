/******************************************************
 COutBitStream Ver.1.1.0 ビットストリーム・クラス
 Copyright 2004 AsWe.Co.Ltd. All rights reserved.
 # The comments are written in Japanese (Shift-JIS).
 ******************************************************/

#include <cassert>
#include <cstring>
#include "out_bit_stream.h"

// 余ったビットに1を詰めるためのマスク
static const byte kBitFullMaskT[8] = { 0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0x7f, 0xff };

OutBitStream::OutBitStream(size_t size) :
	_size(size) {
	assert(size != 0);

	// バッファの設定
	mHeadOfBufP = mBufP = new byte[size];
	// バッファの最終アドレス
	mEndOfBufP = mBufP + size;
	// バッファのクリア
	memset(mHeadOfBufP, 0, size);
	// 状態変数初期化
	mBitPos = 7; // 最上位ビット
	mWriteFlag = 1; // アクセス許可
}

OutBitStream::~OutBitStream() {
	delete[] mHeadOfBufP;
}

void OutBitStream::reset() {
	mBufP = mHeadOfBufP;
	mEndOfBufP = mBufP + _size;
	mBitPos = 7; // 最上位ビット
	mWriteFlag = 1; // アクセス許可
}
void OutBitStream::resize(size_t size) {
	delete[] mHeadOfBufP;

	_size = size;

	// バッファの設定
	mHeadOfBufP = mBufP = new byte[size];

	// バッファの最終アドレス
	mEndOfBufP = mBufP + size;

	// バッファのクリア
	memset(mHeadOfBufP, 0, size);

	// 状態変数初期化
	mBitPos = 7; // 最上位ビット
	mWriteFlag = 1; // アクセス許可
}

// 書き込みアドレスのインクリメントとアクセス違反のチェック
void OutBitStream::incBuf(void) {
	if (++mBufP >= mEndOfBufP)
		mWriteFlag = 0; // 次のアクセスでエラー
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
void OutBitStream::setBits(size_t v, size_t numBits) {
	if (numBits == 0)
		return; // 何もしない

	if (numBits > 8) { // 2バイトなら
		set8Bits(byte(v >> 8), numBits - 8); // 上位バイト書き込み
		numBits = 8;
	}
	set8Bits(byte(v), numBits); // 残り下位バイトを書き込み
}

// 8ビット以下のデータを１つのアドレスに書き込む
void OutBitStream::setFewBits(byte v, size_t numBits) {
	v &= kBitFullMaskT[numBits - 1]; // 上位ビットをクリア
	*mBufP |= v << (mBitPos + 1 - numBits);
	if ((mBitPos -= numBits) < 0) {
		incBuf();
		mBitPos = 7;
	}
}

// 8ビット以下のデータを2つのアドレスに分けて書き込む
void OutBitStream::setBits2Byte(byte v, size_t numBits) {
	v &= kBitFullMaskT[numBits - 1]; // 上位ビットをクリア
	size_t nextBits = numBits - (mBitPos + 1); // 次のバイトに入れるビット数
	*mBufP |= (v >> nextBits) & kBitFullMaskT[mBitPos]; // 1バイト目書き込み

	incBuf(); // 次のアドレス
	*mBufP = v << (8 - nextBits); // 2バイト目書き込み
	mBitPos = 7 - nextBits; // ビット位置更新
}

// 8ビット以下のデータを書き込む
void OutBitStream::set8Bits(byte v, size_t numBits) {
	if (mBitPos + 1 >= numBits) // 現在のバイトに全部入るとき
		setFewBits((byte) v, numBits);
	else
		// 現在のバイトからはみ出すとき
		setBits2Byte((byte) v, numBits);
}

// 余ったビットに1を書き込む
void OutBitStream::fillBit(void) {
	if (mBitPos != 7) // バイトの途中ならば
		setFewBits(kBitFullMaskT[mBitPos], mBitPos + 1); // バイトの終わりまで1を書き込む
}

// 1バイト書き込む
void OutBitStream::setByte(byte dat) {
	if (mWriteFlag) { // 書き込み可
		fillBit(); // ビット書き込み位置の調整
		*mBufP = dat; // 書き込み
		incBuf(); // 書き込みアドレス更新
		return;
	}
}

// 2バイト書き込む
void OutBitStream::setWord(u_int dat) {
	if (mWriteFlag) { // 書き込み可
		fillBit();
		*mBufP = (dat >> 8) & 0xff; // 上位バイト書き込み
		incBuf();
		*mBufP = dat & 0xff; // 下位バイト書き込み
		incBuf();
		return;
	}
}

// nバイト書き込む
void OutBitStream::copyByte(char* srcP, size_t n) {
	if (mBufP + n < mEndOfBufP) { // 書ききれるならば
		fillBit();
		memcpy(mBufP, srcP, n);
		mBufP += n;
		return;
	}
}

