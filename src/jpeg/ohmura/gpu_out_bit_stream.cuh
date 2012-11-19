/******************************************************
 COutBitStream Ver.1.1.0 出力ビットストリーム・クラス
 Copyright 2004 AsWe.Co.Ltd. All right reseved.
 # The comments are written in Japanese (Shift-JIS).
 ******************************************************/
#ifndef GPU_OUT_BIT_STREAM_H_
#define GPU_OUT_BIT_STREAM_H_
#include "../../utils/type_definitions.h"
#include "../../utils/cuda/cuda_memory.hpp"
#include <cstdio>
#include <iostream>

//------------------------------------------------------------
//Max_Block_size:ハフマンエンコードの時のマクロブロック毎のバッファのサイズ,コンスタントのがいいかも
//------------------------------------------------------------
#define MBS 128 //#define MBS 256
namespace jpeg {
	namespace ohmura {

		using namespace util;

		/**
		 * CUDAによる圧縮のMCUごとのステート
		 *
		 * @author yuumomma
		 * @version 1.0
		 */
		class OutBitStreamState {
		public:
			int byte_pos_;
			int bit_pos_; //! 書き込みビット位置（上位ビットが7、下位ビットが0）
			int writable_; //! 1:書き込み可, 0:書き込み不可
			int num_bits_; //! 全体バッファに書き込むサイズ

			OutBitStreamState() :
				byte_pos_(0),
				bit_pos_(7), // 最上位ビット
				writable_(1), // アクセス許可
				num_bits_(0) {
			}

			/**
			 * ストリーム出力
			 * @param os
			 * @param state
			 * @return
			 */
			friend std::ostream& operator<<(std::ostream& os, const OutBitStreamState& state) {
				return (os << "{\n" << "_byte_pos	:" << state.byte_pos_ << "\n" << "_bit_pos	:"
					<< state.bit_pos_ << "\n" << "_writable	:" << state.writable_ << "\n"
					<< "_num_bits	:" << state.num_bits_ << "\n" << "}\n");
			}
		};

		/**
		 * 出力バッファ
		 *
		 * @author yuumomma
		 * @version 1.0
		 */
		class OutBitStreamBuffer {
		private:
			util::cuda::cuda_memory<byte> head_of_buff_; //! バッファの先頭アドレス
			byte* write_buff_address_; //! 書き込みアドレス
			byte* end_of_buff_; //! バッファの終了アドレス
			size_t buff_size_;

		public:
			/**
			 * コンストラクタ
			 * @param buff_size バッファサイズ
			 */
			OutBitStreamBuffer(size_t buff_size) :
				buff_size_(buff_size),
				head_of_buff_(buff_size) {

				head_of_buff_.fill_zero();
				write_buff_address_ = head_of_buff_.device_data();
				end_of_buff_ = head_of_buff_.device_data() + buff_size_; // バッファの最終アドレス
			}

			/**
			 * リサイズする
			 * @param size 変更サイズ
			 * @param force より小さくする際に、現在のバッファを完全に破棄するかどうか
			 */
			void resize(size_t size, bool force = false) {
				buff_size_ = size;
				head_of_buff_.resize(size, force);

				head_of_buff_.fill_zero();
				write_buff_address_ = head_of_buff_.device_data();
				end_of_buff_ = head_of_buff_.device_data() + buff_size_; // バッファの最終アドレス
			}

			/**
			 * CUDAバッファを取得する
			 * @return CUDAメモリ
			 */
			util::cuda::cuda_memory<byte>& get_stream_buffer() {
				return head_of_buff_;
			}
			/**
			 * CUDAバッファを取得する
			 * @return CUDAメモリ
			 */
			const util::cuda::cuda_memory<byte>& get_stream_buffer() const {
				return head_of_buff_;
			}

			/**
			 * デバイスメモリの先頭を取得する
			 * @return デバイスメモリの先頭
			 */
			byte* head_device() {
				return write_buff_address_;
			}
			/**
			 * デバイスメモリの先頭を取得する
			 * @return デバイスメモリの先頭
			 */
			const byte* head_device() const {
				return write_buff_address_;
			}

			/**
			 * デバイスメモリの末尾を取得する
			 * @return デバイスメモリの末尾
			 */
			byte* end_device() {
				return end_of_buff_;
			}
			/**
			 * デバイスメモリの末尾を取得する
			 * @return デバイスメモリの末尾
			 */
			const byte* end_device() const {
				return end_of_buff_;
			}
		};
		/** 余ったビットに1を詰めるためのマスク */__device__
		           __constant__           static const byte kBitFullMaskT[8] = {
			0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0x7f, 0xff };
		/** 余ったビットに1を詰めるためのマスク */__device__
		           __constant__           static const byte kBitFullMaskLowT[8] = {
			0x80, 0xc0, 0xe0, 0xf0, 0xf8, 0xfc, 0xfe, 0xff };

		/**
		 * バッファのカウンタをすすめる
		 * @param state マクロブロック毎の書き込み位置と書き込みbit数を記録してある
		 * @param dst 書き込み先先頭アドレス、一枚のバッファ
		 * @param end_of_buff バッファの末尾
		 */
		inline __device__ void IncBuf(OutBitStreamState *state, byte *dst, byte *end_of_buff) {
			//エラー検出、ちょっとだけ遅くなる
			if (++state->byte_pos_ >= MBS) {
				printf("IncBuf:buff_overflow");
			}
		}

		/**
		 * 8ビット以下のデータを１つのアドレスに書き込む
		 * @param state マクロブロック毎の書き込み位置と書き込みbit数を記録してある
		 * @param dst 書き込み先先頭アドレス、一枚のバッファ
		 * @param end_of_buff バッファの末尾
		 * @param value 書き込む値
		 * @param num_bits 書き込みビット数
		 */
		inline __device__ void SetFewBits(OutBitStreamState *state, byte *dst, byte *end_of_buff,
			byte value, int num_bits) {
			// 上位ビットをクリア
			value &= kBitFullMaskT[num_bits - 1];
			*(dst + state->byte_pos_) |= value << (state->bit_pos_ + 1 - num_bits);
			if ((state->bit_pos_ -= num_bits) < 0) {
				IncBuf(state, dst, end_of_buff);
				state->bit_pos_ = 7;
			}
		}

		/**
		 * 8ビット以下のデータを2つのアドレスに分けて書き込む
		 * @param state マクロブロック毎の書き込み位置と書き込みbit数を記録してある
		 * @param dst 書き込み先先頭アドレス、一枚のバッファ
		 * @param end_of_buff バッファの末尾
		 * @param value 書き込む値
		 * @param num_bits 書き込みビット数
		 */
		inline __device__ void SetBits2Byte(OutBitStreamState *state, byte *dst, byte *end_of_buff,
			byte value, int num_bits) {
			value &= kBitFullMaskT[num_bits - 1]; // 上位ビットをクリア
			int nextBits = num_bits - (state->bit_pos_ + 1); // 次のバイトに入れるビット数
			*(dst + state->byte_pos_) |= (value >> nextBits) & kBitFullMaskT[state->bit_pos_]; // 1バイト目書き込み

			IncBuf(state, dst, end_of_buff);
			*(dst + state->byte_pos_) |= value << (8 - nextBits); // 2バイト目書き込み
			state->bit_pos_ = 7 - nextBits; // ビット位置更新
		}

		/**
		 * 8ビット以下のデータを書き込む
		 * @param state マクロブロック毎の書き込み位置と書き込みbit数を記録してある
		 * @param dst 書き込み先先頭アドレス、一枚のバッファ
		 * @param end_of_buff バッファの末尾
		 * @param value 書き込む値
		 * @param num_bits 書き込みビット数
		 */
		inline __device__ void Set8Bits(OutBitStreamState *state, byte *dst, byte *end_of_buff,
			byte value, int num_bits) {
			if (state->bit_pos_ + 1 >= num_bits) // 現在のバイトに全部入るとき
				SetFewBits(state, dst, end_of_buff, (byte) value, num_bits);
			else
				// 現在のバイトからはみ出すとき
				SetBits2Byte(state, dst, end_of_buff, (byte) value, num_bits);
		}

		/**
		 * ビット単位で書き込む（最大16ビット）
		 *
		 * - valueには右詰めでデータが入っている。
		 * - dstには左詰めで格納する
		 * - valueが負の時下位ビットのみを格納する
		 * - num_bitsは1以上16以下
		 *
		 * value = 00010011(2)のとき10011を*mBufPに追加する
		 *
		 * @param state マクロブロック毎の書き込み位置と書き込みbit数を記録してある
		 * @param dst 書き込み先先頭アドレス、一枚のバッファ
		 * @param end_of_buff バッファの末尾
		 * @param value 書き込む値
		 * @param num_bits 書き込みビット数
		 */
		inline __device__ void SetBits(OutBitStreamState *state, byte *dst, byte *end_of_buff,
			int value, int num_bits) {
			if (num_bits == 0)
				return;

			if (num_bits > 8) { // 2バイトなら
				Set8Bits(state, dst, end_of_buff, byte(value >> 8), num_bits - 8); // 上位バイト書き込み
				num_bits = 8;
			}
			Set8Bits(state, dst, end_of_buff, byte(value), num_bits); // 残り下位バイトを書き込み
		}

		//==============================================================================================
		// MCU毎から1枚のバッファへ。上位ビットから書き込むので上の流用不可
		//==============================================================================================
		/**
		 * バッファのカウンタをすすめる
		 * @param state マクロブロック毎の書き込み位置と書き込みbit数を記録してある
		 */
		inline __device__ void IncBuf_w(OutBitStreamState *state) {
			++state->byte_pos_;
		}

		/**
		 * 8ビット以下のデータを１つのアドレスに書き込む
		 * @param state マクロブロック毎の書き込み位置と書き込みbit数を記録してある
		 * @param dst 書き込み先先頭アドレス、一枚のバッファ
		 * @param value 書き込む値
		 * @param num_bits 書き込みビット数
		 */
		inline __device__ void SetFewBits_w(OutBitStreamState *state, byte *dst, byte value, // 書き込む値
			int num_bits) // 書き込みビット数
			{
			value &= kBitFullMaskLowT[num_bits - 1]; // 下位ビットをクリア
			*(dst + state->byte_pos_) |= (value >> 7 - state->bit_pos_); //

			if ((state->bit_pos_ -= num_bits) < 0) {
				//マーカが存在しないのでいらない、デコーダも修正

				IncBuf_w(state);
				state->bit_pos_ = 7;
			}
		}

		/**
		 * 8ビット以下のデータを2つのアドレスに分けて書き込む
		 * @param state マクロブロック毎の書き込み位置と書き込みbit数を記録してある
		 * @param dst 書き込み先先頭アドレス、一枚のバッファ
		 * @param value 書き込む値
		 * @param num_bits 書き込みビット数
		 */
		inline __device__ void SetBits2Byte_w(OutBitStreamState *state, byte *dst, byte value, // 書き込む値
			int num_bits) // 書き込みビット数
			{
			value &= kBitFullMaskLowT[num_bits - 1]; // 下位ビットをクリア
			int nextBits = num_bits - (state->bit_pos_ + 1); // 次のバイトに入れるビット数

			*(dst + state->byte_pos_) |= (value >> 7 - state->bit_pos_); // 1バイト目書き込み

			IncBuf_w(state);
			*(dst + state->byte_pos_) |= (value << state->bit_pos_ + 1); // 2バイト目書き込み
			state->bit_pos_ = 7 - nextBits; // ビット位置更新
		}

		/**
		 * 8ビット以下のデータを書き込む
		 * @param state マクロブロック毎の書き込み位置と書き込みbit数を記録してある
		 * @param dst 書き込み先先頭アドレス、一枚のバッファ
		 * @param value 書き込む値
		 * @param num_bits 書き込みビット数
		 */
		inline __device__ void Set8Bits_w(OutBitStreamState *state, byte *dst, byte value, // 書き込む値
			int num_bits) // 書き込みビット数
			{
			if (state->bit_pos_ + 1 >= num_bits) // 現在のバイトに全部入るとき
				SetFewBits_w(state, dst, (byte) value, num_bits);
			else
				// 現在のバイトからはみ出すとき
				SetBits2Byte_w(state, dst, (byte) value, num_bits);
		}

		/**
		 * bitを書き込む
		 *
		 * @param state マクロブロック毎の書き込み位置と書き込みbit数を記録してある
		 * @param dst 書き込み先先頭アドレス、一枚のバッファ
		 * @param src 書き込み元先頭アドレス、マクロ毎のバッファ
		 */
		inline __device__ void WriteBits(OutBitStreamState *Od, //マクロブロック毎の書き込み位置と書き込みbit数を記録してある
			byte *OmBufP, //書き込み先先頭アドレス、一枚のバッファ
			byte *ImBufP, // 書き込み元先頭アドレス、マクロ毎のバッファ
			int id //バグチェック用に使ってた
			) {
			int bytepos = 0;
			while (Od->num_bits_ > 8) {
				Set8Bits_w(Od, OmBufP, *(ImBufP + bytepos), 8); // 1バイト書き込み
				Od->num_bits_ -= 8;
				bytepos++;
			}

			Set8Bits_w(Od, OmBufP, *(ImBufP + bytepos), Od->num_bits_); // 端数バイト書き込み
		}
	} // namespace gpu
} // namespace jpeg
#endif
