/******************************************************
 COutBitStream Ver.1.1.0 出力ビットストリーム・クラス
 Copyright 2004 AsWe.Co.Ltd. All right reseved.
 # The comments are written in Japanese (Shift-JIS).
 ******************************************************/

#ifndef OUT_BIT_STREAM_H_
#define OUT_BIT_STREAM_H_

#include <utils/type_definitions.h>

namespace util {
	/**
	 * @brief ビット出力クラス
	 *
	 * @author AsWe.Co midified by yuumomma
	 * @version 1.0
	 */
	class OutBitStream {
	public:
		/**
		 * @brief コンストラクタ
		 *
		 * @param size バッファのサイズ(byte)
		 */
		OutBitStream(size_t size);

		/**
		 * @brief デストラクタ
		 */
		~OutBitStream();

		/**
		 * @brief ビット単位で書き込む（最大16ビット）
		 *
		 * @param v 書き込む値
		 * @param numBits 書き込むビット数
		 */
		void setBits(size_t v, size_t numBits);

		/**
		 * @brief 1バイト書き込む
		 *
		 * @param dat 書き込む値
		 */
		void setByte(byte dat);

		/**
		 * @brief 2バイト書き込む
		 *
		 * @param dat 書き込む値
		 */
		void setWord(u_int dat);

		/**
		 * @brief nバイト書き込む
		 *
		 * @param srcP 書き込む値
		 * @param n 書き込むバイト数
		 */
		void copyByte(char* srcP, size_t n);

		/**
		 * @brief ストリームの先頭アドレス
		 *
		 * @return 先頭アドレス
		 */
		byte* getStreamAddress(void) {
			return mHeadOfBufP;
		}

		/**
		 * @brief ストリームの有効サイズ
		 *
		 * @return 有効サイズ
		 */
		size_t getStreamSize(void) {
			return size_t(mBufP - mHeadOfBufP);
		}

		/**
		 * @brief すべての書き込み内容を破棄し、初期状態に戻す
		 */
		void reset();

		/**
		 * @brief すべての書き込み内容を破棄し、バッファサイズを変更する
		 *
		 * @param size 変更サイズ
		 */
		void resize(size_t size);

	private:
		byte* mHeadOfBufP; 	//! バッファの先頭アドレス
		byte* mBufP; 		//! 書き込みアドレス
		byte* mEndOfBufP; 	//! バッファの終了アドレス
		size_t mBitPos; 	//! 書き込みビット位置（上位ビットが7、下位ビットが0）
		size_t mWriteFlag; 	//! 1:書き込み可, 0:書き込み不可
		size_t _size;

		/** @brief 書き込みアドレスのインクリメントとアクセス違反のチェック */
		void incBuf(void);
		/** @brief 余ったビットに1を書き込む */
		void fillBit(void);
		/** @brief 8ビット以下のデータを書き込む */
		void set8Bits(byte v, size_t numBits);
		/** @brief 端数ビットを書き込む */
		void setFewBits(byte v, size_t numBits);
		/** @brief 8ビット以下のデータを2つのアドレスに分けて書き込む */
		void setBits2Byte(byte v, size_t numBits);

		OutBitStream(OutBitStream &);
		void operator =(OutBitStream &);
	};
}  // namespace util

#endif /* OUT_BIT_STREAM_H_ */
