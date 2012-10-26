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
	OutBitStream(size_t size);
	~OutBitStream();

	// ビット単位で書き込む（最大16ビット）
	void setBits(size_t v, size_t numBits);

	// 1バイト書き込む
	void setByte(byte dat);

	// 2バイト書き込む
	void setWord(u_int dat);

	// nバイト書き込む
	void copyByte(char* srcP, size_t n); 

	// ストリームの先頭アドレス
	byte* getStreamAddress(void) {
		return mHeadOfBufP;
	}

	// ストリームの有効サイズ
	size_t getStreamSize(void) {
		return size_t(mBufP - mHeadOfBufP);
	}

	void reset();
	void resize(size_t size);

private:
	byte* mHeadOfBufP; // バッファの先頭アドレス
	byte* mBufP; // 書き込みアドレス
	byte* mEndOfBufP; // バッファの終了アドレス
	size_t mBitPos; // 書き込みビット位置（上位ビットが7、下位ビットが0）
	size_t mWriteFlag; // 1:書き込み可, 0:書き込み不可
	size_t _size;

	// 書き込みアドレスのインクリメントとアクセス違反のチェック
	void incBuf(void);
	// 余ったビットに1を書き込む
	void fillBit(void);
	// 8ビット以下のデータを書き込む
	void set8Bits(byte v, size_t numBits);
	// 端数ビットを書き込む
	void setFewBits(byte v, size_t numBits);
	// 8ビット以下のデータを2つのアドレスに分けて書き込む
	void setBits2Byte(byte v, size_t numBits);

	OutBitStream(OutBitStream &);
	void operator =(OutBitStream &);
};

#endif /* OUT_BIT_STREAM_H_ */
