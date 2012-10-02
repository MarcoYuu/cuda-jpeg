/******************************************************
 COutBitStream Ver.1.1.0 出力ビットストリーム・クラス
 Copyright 2004 AsWe.Co.Ltd. All right reseved.
 # The comments are written in Japanese (Shift-JIS).
 ******************************************************/

#ifndef OUT_BIT_STREAM_H_
#define OUT_BIT_STREAM_H_

#include "../type_definitions.h"

class OutBitStream {
public:
	// バッファのサイズ(byte)
	OutBitStream(int size);

	// ビット単位で書き込む（最大16ビット）
	void SetBits(int v, int numBits);

	// 1バイト書き込む
	void SetByte(byte dat);

	// 2バイト書き込む
	void SetWord(u_int dat);

	// nバイト書き込む
	void CopyByte(char* srcP, // ソースアドレス
	int n); // 書き込みバイト数

	// ストリームの先頭アドレス
	byte* GetStreamAddress(void) {
		return mHeadOfBufP;
	}

	// ストリームの有効サイズ
	int GetStreamSize(void) {
		return int(mBufP - mHeadOfBufP);
	}

protected:
	byte* mHeadOfBufP; // バッファの先頭アドレス
	byte* mBufP; // 書き込みアドレス
	byte* mEndOfBufP; // バッファの終了アドレス
	int mBitPos; // 書き込みビット位置（上位ビットが7、下位ビットが0）
	int mWriteFlag; // 1:書き込み可, 0:書き込み不可

	// 書き込みアドレスのインクリメントとアクセス違反のチェック
	void IncBuf(void);
	// 余ったビットに1を書き込む
	void FullBit(void);
	// 8ビット以下のデータを書き込む
	void Set8Bits(byte v, int numBits);
	// 端数ビットを書き込む
	void SetFewBits(byte v, int numBits);
	// 8ビット以下のデータを2つのアドレスに分けて書き込む
	void SetBits2Byte(byte v, int numBits);
};

#endif /* OUT_BIT_STREAM_H_ */
