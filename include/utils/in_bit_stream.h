// ******************************************************
// CInBitStream Ver.1.0.0 ビットストリーム・クラス
// Copyright 2004 AsWe.Co.,Ltd. All rights reserved.
// # The comments are written in Japanese (Shift-JIS).
// ******************************************************

#ifndef IN_BIT_STREAM_H_
#define IN_BIT_STREAM_H_

#include <utils/type_definitions.h>

namespace util {
	/**
	 * @brief ビット読み出しクラス
	 *
	 * コンストラクタに渡した読み込みバッファは、
	 * このクラスの寿命より先に破棄されてはいけない
	 *
	 * @author AsWe.Co modified by momma
	 * @version 1.0
	 */
	class InBitStream {
	public:
		/**
		 * @brief コンストラクタ
		 *
		 * @param aBufP 読み込みバッファ
		 * @param size バッファの有効サイズ
		 */
		InBitStream(const byte* aBufP, size_t size);

		/**
		 *
		 * @param aBufP 読み込みバッファ
		 * @param size バッファの有効サイズ
		 */
		void reset(const byte* aBufP, size_t size);

		/**
		 * @brief ビット単位で読み出す
		 *
		 * @param numOfBits 読みだすビット数
		 * @return 読み出し値
		 */
		int getBits(size_t numOfBits);

		/** @brief 1バイト読み出す */
		byte getByte();

		/** @brief 2バイト読み出す */
		u_int getWord();

		/**
		 * @brief nバイト読み出す
		 *
		 * @param disP [out] 結果を読み出すバッファ
		 * @param n 読みだすビット数
		 */
		void copyByte(byte* disP, size_t n);

		/**
		 * @brief nバイト進める
		 *
		 * @param n 読み飛ばすバイト数
		 */
		void skipByte(size_t n);

		/**
		 * @brief 次の読み出しアドレスを取得する
		 *
		 * @return アドレス
		 */
		char* getNextAddress();

	private:
		byte* mBufP; 		//! 読み出しアドレス
		byte* mEndOfBufP; 	//! バッファの終了アドレス
		int mBitPos; 		//! 読み出しビット位置（上位ビットが7、下位ビットが0）
		int mNextFlag; 		//! 次のバイトを読んでいいかどうか
		int mReadFlag; 		//! 1:読み出し可, 0:読み出し不可

		/** @brief 読み出しアドレスのインクリメントとアクセス違反のチェック */
		void incBuf();

		InBitStream(InBitStream &);
		void operator =(InBitStream &);
	};
}  // namespace util

#endif /* IN_BIT_STREAM_H_ */
