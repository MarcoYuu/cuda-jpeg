/******************************************************
 CInBitStream ビットストリーム・クラス Ver.1.1.0
 Copyright 2004 AsWe Co.,Ltd.
 All rights reserved.
 The comments are written in Japanese (Shift-JIS).
 ******************************************************/

#include <cassert>
#include <cstring>

#include <utils/in_bit_stream.h>

namespace util {
	// ビット取り出しのためのマスク
	static const int kBitTestMaskT[8] = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 };

	InBitStream::InBitStream(const byte* aBufP, size_t size) {
		// データバッファの設定
		mBufP = (byte*) aBufP; // バッファ
		mEndOfBufP = mBufP + size; // バッファの最終アドレス

		// 状態変数初期化
		mBitPos = 7; // 最上位ビット
		mNextFlag = 1; // 読み飛ばし無し
		mReadFlag = 1; // アクセスエラー無し
	}

	void InBitStream::reset(const byte* aBufP, size_t size) {
		// データバッファの設定
		mBufP = (byte*) aBufP; // バッファ
		mEndOfBufP = mBufP + size; // バッファの最終アドレス

		// 状態変数初期化
		mBitPos = 7; // 最上位ビット
		mNextFlag = 1; // 読み飛ばし無し
		mReadFlag = 1; // アクセスエラー無し
	}

	// 読み出しアドレスのインクリメントとアクセス違反のチェック
	void InBitStream::incBuf() {
		if (++mBufP >= mEndOfBufP) // 次のアクセスでエラー
			mReadFlag = 0;
	}

	// ビット単位で読み出す
	int InBitStream::getBits(size_t numOfBits) {
		if (numOfBits <= 0)
			return 0; // エラー

		int r = 0; // 返値
		while (numOfBits) {
			if (mBitPos < 0) { // 次のバイトを読み出すとき
				mBitPos = 7; // 読み出しビット位置更新
				incBuf(); // アドレス更新
			}

			// 返値の作成
			r <<= 1;
			r |= ((*mBufP) & kBitTestMaskT[mBitPos--]) ? 1 : 0;
			// 1ビット読み出しと読み出しビット位置更新
			numOfBits--; // 読み出しビット数更新
		}
		return r;
	}

	// 1バイト読み出す
	byte InBitStream::getByte() {
		//if (mReadFlag) {
		assert(mReadFlag);
		if (mBitPos != 7) { // 読み出し位置がバイトの途中ならば
			incBuf(); // 次のバイトの
			mBitPos = 7; // 最初のビットから
		}
		byte r = *mBufP; // 1バイト読み出し
		incBuf(); // 読み出し位置更新
		mNextFlag = 1; // ビット読み出し中にマーカを検出した場合0になるが、
		// GetByte()でマーカを処理するので次のバイトを読んでよい
		return r;
		//}
	}

	// 2バイト読み出す
	u_int InBitStream::getWord() {
		assert(mReadFlag);
		if (mBitPos != 7) { // バイトの途中から読み出さない
			incBuf();
			mBitPos = 7;
		}
		u_int r = (unsigned) *mBufP << 8;
		incBuf();
		r |= (unsigned) *mBufP;
		incBuf();

		return r;
	}

	// nバイト読み出す
	void InBitStream::copyByte(byte* disP, size_t n) {
		if (mBitPos != 7) { // バイトの途中から読み出さない
			incBuf();
			mBitPos = 7;
		}

		memcpy(disP, mBufP, n);
		if ((mBufP += n) >= mEndOfBufP) // 次のアクセスでエラー
			mReadFlag = 0;
	}

	// nバイト進める
	void InBitStream::skipByte(size_t n) {
		if ((mBufP += n) >= mEndOfBufP) // 次のアクセスでエラー
			mReadFlag = 0;
		mBitPos = 7;
	}

	// 次の読み出しアドレス
	char* InBitStream::getNextAddress() {
		if (mBitPos == 7) // 読み出し途中でないなら
			return (char*) mBufP;
		else if (mBufP < mEndOfBufP)
			return (char*) (mBufP + 1);
		else
			return NULL;
	}
}  // namespace util
