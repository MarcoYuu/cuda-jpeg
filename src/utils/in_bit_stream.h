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
	InBitStream(char* aBufP, int size);

	// ビット単位で読み出す
	int GetBits(int numOfBits);

	// 1バイト読み出す
	byte GetByte(void);

	// 2バイト読み出す
	u_int GetWord(void);

	// nバイト読み出す
	void CopyByte(char* disP, int n);

	// nバイト進める
	void SkipByte(int n);

	// 次の読み出しアドレス
	char* GetNextAddress(void);

protected:
	byte* mBufP; // 読み出しアドレス
	byte* mEndOfBufP; // バッファの終了アドレス
	int mBitPos; // 読み出しビット位置（上位ビットが7、下位ビットが0）
	int mNextFlag; // 次のバイトを読んでいいかどうか
	int mReadFlag; // 1:読み出し可, 0:読み出し不可

	// 読み出しアドレスのインクリメントとアクセス違反のチェック
	void IncBuf(void);
};

#endif /* IN_BIT_STREAM_H_ */
