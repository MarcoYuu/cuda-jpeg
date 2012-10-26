// ******************************************************
// CInBitStream Ver.1.0.0 ビットストリーム・クラス
// Copyright 2004 AsWe.Co.,Ltd. All rights reserved.
// # The comments are written in Japanese (Shift-JIS).
// ******************************************************

#ifndef IN_BIT_STREAM_H_
#define IN_BIT_STREAM_H_

#include "../type_definitions.h"

class InBitStream {
public:
	// コンストラクタ
	InBitStream(const byte* aBufP, size_t size);

	// ビット単位で読み出す
	int getBits(size_t numOfBits);

	// 1バイト読み出す
	byte getByte();

	// 2バイト読み出す
	u_int getWord();

	// nバイト読み出す
	void copyByte(byte* disP, size_t n);

	// nバイト進める
	void skipByte(size_t n);

	// 次の読み出しアドレス
	char* getNextAddress();

private:
	byte* mBufP; // 読み出しアドレス
	byte* mEndOfBufP; // バッファの終了アドレス
	int mBitPos; // 読み出しビット位置（上位ビットが7、下位ビットが0）
	int mNextFlag; // 次のバイトを読んでいいかどうか
	int mReadFlag; // 1:読み出し可, 0:読み出し不可

	// 読み出しアドレスのインクリメントとアクセス違反のチェック
	void incBuf();

	InBitStream(InBitStream &);
	void operator =(InBitStream &);
};

#endif /* IN_BIT_STREAM_H_ */
