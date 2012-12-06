#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <jpeg/cpu/cpu_jpeg.h>
#include <jpeg/cuda/cuda_jpeg.cuh>

#include <utils/debug_log.h>
#include <utils/utility.hpp>
#include <utils/cuda/bit_operation.cuh>

#include "encoder_tables_device.cuh"

namespace jpeg {
	namespace cuda {

		using namespace util;
		using namespace util::cuda;
		using namespace encode_table;

		using util::u_int;

		//-------------------------------------------------------------------------------------------------//
		//
		// ハフマン符号化
		//
		//-------------------------------------------------------------------------------------------------//
		namespace kernel {
			using namespace cuda::encode_table::HuffmanEncode;

			/** @brief 余ったビットに1を詰めるためのマスク */
			static __device__ __constant__ const unsigned char kBitFullMaskT[] = {
				0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0x7f, 0xff };
			/** @brief 余ったビットに1を詰めるためのマスク */
			static __device__ __constant__ const unsigned char kBitFullMaskLowT[] = {
				0xff, 0x7f, 0x3f, 0x1f, 0x0f, 0x07, 0x03, 0x01 };

			/**
			 * @brief CUDAによる圧縮のMCUごとのステート
			 *
			 * @author yuumomma
			 * @version 1.0
			 */
			class OutBitStream {

				u_int byte_pos_;
				int bit_pos_; /// 書き込みビット位置（上位ビットが7、下位ビットが0）
				u_int writable_; /// 1:書き込み可, 0:書き込み不可
				byte *dst_buf_; ///

			public:
				static const u_int MAX_BLOCK_SIZE = 128;

				/**
				 * @brief コンストラクタ
				 */
				inline __host__ OutBitStream() :
					byte_pos_(0),
					bit_pos_(7),
					writable_(1),
					dst_buf_(NULL) {
				}

				/**
				 * @brief コンストラクタ
				 *
				 * 必ずMAX_BLOCK_SIZE以上のサイズのデバイスメモリを指定すること
				 *
				 * @param device_buf デバイスメモリへのポインタ
				 */
				inline __host__ OutBitStream(byte* device_buf) :
					byte_pos_(0),
					bit_pos_(7),
					writable_(1),
					dst_buf_(device_buf) {
				}

				/**
				 * @brief ストリームに関連付けられるバッファを設定
				 *
				 * 必ずMAX_BLOCK_SIZE以上のサイズのデバイスメモリを指定すること
				 *
				 * @param device_buf デバイスメモリへのポインタ
				 */
				inline __host__ void setStreamBuffer(byte* device_buf) {
					dst_buf_ = device_buf;
				}

				/**
				 * @brief 関連付けられたデータの取得
				 * @return データバッファ
				 */
				inline __device__ byte* getData() {
					return dst_buf_;
				}

				/**
				 * @brief 関連付けられたデータの取得
				 * @return データバッファ
				 */
				inline const __device__ byte* getData() const {
					return dst_buf_;
				}

				/**
				 * @brief ビット単位で書き込む（最大16ビット）
				 *
				 * - valueには右詰めでデータが入っている。
				 * - dstには左詰めで格納する
				 * - valueが負の時下位ビットのみを格納する
				 * - num_bitsは1以上16以下
				 *
				 * value = 00010011(2)のとき10011を*mBufPに追加する
				 *
				 * @param value 書き込む値
				 * @param num_bits 書き込みビット数
				 */
				inline __device__ void setBits(u_int value, u_int num_bits) {
					assert(num_bits != 0);

					if (num_bits > 8) { // 2バイトなら
						set8Bits(byte(value >> 8), num_bits - 8); // 上位バイト書き込み
						num_bits = 8;
					}
					set8Bits(byte(value), num_bits); // 残り下位バイトを書き込み
				}

			private:
				/**
				 * @brief バッファのカウンタをすすめる
				 */
				inline __device__ void increment() {
					++byte_pos_;
					assert((byte_pos_ < MAX_BLOCK_SIZE));
				}

				/**
				 * @brief 8ビット以下のデータを１つのアドレスに書き込む
				 *
				 * @param value 書き込む値
				 * @param num_bits 書き込みビット数
				 */
				inline __device__ void setFewBits(byte value, u_int num_bits) {
					// 上位ビットをクリア
					value &= kBitFullMaskT[num_bits - 1];

					*(dst_buf_ + byte_pos_) |= value << (bit_pos_ + 1 - num_bits);

					bit_pos_ -= num_bits;
					if (bit_pos_ < 0) {
						increment();
						bit_pos_ = 7;
					}
				}

				/**
				 * @brief 8ビット以下のデータを2つのアドレスに分けて書き込む
				 *
				 * @param value 書き込む値
				 * @param num_bits 書き込みビット数
				 */
				inline __device__ void setBits2Byte(byte value, u_int num_bits) {
					// 上位ビットをクリア
					value &= kBitFullMaskT[num_bits - 1];
					// 次のバイトに入れるビット数
					u_int nextBits = num_bits - (bit_pos_ + 1);

					// 1バイト目書き込み
					*(dst_buf_ + byte_pos_) |= (value >> nextBits) & kBitFullMaskT[bit_pos_];
					increment();
					// 2バイト目書き込み
					*(dst_buf_ + byte_pos_) |= value << (8 - nextBits);

					// ビット位置更新
					bit_pos_ = 7 - nextBits;
				}

				/**
				 * @brief 8ビット以下のデータを書き込む
				 *
				 * @param value 書き込む値
				 * @param num_bits 書き込みビット数
				 */
				inline __device__ void set8Bits(byte value, u_int num_bits) {
					// 現在のバイトに全部入るとき
					if (bit_pos_ + 1 >= num_bits) {
						setFewBits(value, num_bits);
					}
					// 現在のバイトからはみ出すとき
					else {
						setBits2Byte(value, num_bits);
					}
				}

			public:
				/**
				 * @brief ハフマンバッファの結合情報
				 *
				 * @author yuumomma
				 * @version 1.0
				 */
				struct WriteBitsInfo {
					u_int bits_of_stream; /// このバッファのbit数
					u_int bits_of_grobal; /// このバッファまでのbit数
				};
				typedef std::vector<WriteBitsInfo> WriteBitsTable;

				/**
				 * @brief ハフマンバッファの結合情報の作成
				 *
				 * @param stream ハフマン符号ストリームの配列へのポインタ
				 * @param num_bits 結合情報配列へのポインタ
				 * @param stream_num ストリーム配列のサイズ
				 * @param stream_per_block ブロックを構成するストリームの個数
				 */
				static __host__ void CreateWriteBitsTable(const OutBitStream *stream,
					WriteBitsInfo* num_bits, u_int stream_num, u_int stream_per_block) {

					u_int blocks = stream_num / stream_per_block;

					for (u_int block_index = 0; block_index < blocks; ++block_index) {
						//BitPosは7が上位で0が下位なので注意,更に位置なので注意。7なら要素は0,0なら要素は7
						u_int first_stream_of_block = block_index * stream_per_block;

						num_bits[first_stream_of_block].bits_of_stream = stream[first_stream_of_block].byte_pos_ * 8
							+ (7 - stream[first_stream_of_block].bit_pos_);
						num_bits[first_stream_of_block].bits_of_grobal = 0;

						for (u_int j = first_stream_of_block + 1;
							j < first_stream_of_block + stream_per_block; ++j) {
							num_bits[j].bits_of_stream = stream[j].byte_pos_ * 8
								+ (7 - stream[j].bit_pos_);
							num_bits[j].bits_of_grobal = num_bits[j - 1].bits_of_stream
								+ num_bits[j - 1].bits_of_grobal;
						}
					}
				}

				/**
				 * @brief ストリームの結合
				 *
				 * ストリームに含まれる有効bitを一つのバッファに結合する
				 *
				 * @param stream 結合するストリーム
				 * @param info ストリームの結合情報
				 * @param dst 書き込み先バッファの先頭ポインタ
				 */
				static __device__ void WriteStreamToLineBuffer(const OutBitStream *stream,
					const WriteBitsInfo *info, byte *dst) {
					const u_int bytes = info->bits_of_stream / 8;
					const u_int rest_bits = info->bits_of_stream - 8 * bytes;

					const u_int offset_bytes = info->bits_of_grobal / 8;
					const u_int offset_bits = info->bits_of_grobal - 8 * offset_bytes;

					const byte *data = stream->getData(); // [0, OutBitStream::MAX_BLOCK_SIZE(128)]
					for (u_int i = 0; i < bytes; ++i) {
						dst[offset_bytes + i] |= ((data[i] >> offset_bits) & kBitFullMaskLowT[offset_bits]);
						dst[offset_bytes + i + 1] |= (data[i] << (8 - offset_bits));
					}
					dst[offset_bytes + bytes] |= ((data[bytes] >> offset_bits) & kBitFullMaskLowT[offset_bits]
						& ~((byte) 0xff >> (offset_bits + rest_bits)));
					dst[offset_bytes + bytes + 1] |= (data[bytes] << (8 - offset_bits));
				}
			};

			/** @brief DC成分用コードサイズテーブルのテーブル */
			static __device__ __constant__ const unsigned int* DCSizeTable[] = {
				DC::luminance::code_size, DC::luminance::code_size, DC::component::code_size };

			/** @brief DC成分用ハフマンテーブルのテーブル */
			static __device__ __constant__ const unsigned int* DCCodeTable[] = {
				DC::luminance::code, DC::luminance::code, DC::component::code };

			/** @brief AC成分用コードサイズテーブルのテーブル */
			static __device__ __constant__ const unsigned int* ACSizeTable[] = {
				AC::luminance::code_size, AC::luminance::code_size, AC::component::code_size };

			/** @brief AC成分用ハフマンテーブルのテーブル */
			static __device__ __constant__ const unsigned int* ACCodeTable[] = {
				AC::luminance::code, AC::luminance::code, AC::component::code };

			/**
			 * @brief MCU(8x8)毎ハフマン符号化カーネル
			 *
			 * - 基本的に[huffman_code][value][huffman_code][value]…と続く
			 * - カーネル呼び出しは8x8ブロックごと = (width*height*3/2)/64 thread = mcu_num.
			 * - 最低は16x16の画像で、8X8ブロックは6個 = 最低 16x16x3/2/64 : 6 thread.
			 * - blockIDx.yでわけるとするとTHREADSは最大 buffer_size/64/6/block_num.
			 *
			 *	- buffer_size = width * height * 3 / 2;
			 * 	- block_size = block_width * block_height * 3 / 2;
			 * 	- block_num = buffer_size / block_size;
			 * 	- THREADS = block_size / 64 / 6;
			 *
			 * 	- grid(block_size / 64 / 6 / THREADS, 6, block_num)
			 * 	- block(THREADS, 1, 1)
			 *
			 * @param quantized 量子化データ
			 * @param dst 書き込み先
			 *
			 */__global__ void HuffmanEncodeForMCU(const int *quantized, OutBitStream *dst) {
				using namespace encode_table::HuffmanEncode;

				// マクロブロック番号
				const u_int mcu_id = threadIdx.x + blockIdx.x * blockDim.x
					+ blockIdx.y * blockDim.x * gridDim.x + blockIdx.z * blockDim.x * gridDim.x * gridDim.y;

				// 量子化結果におけるマクロブロックの開始インデックス
				const u_int mcu_start_index = 64 * mcu_id;

				// マクロブロックごとのバッファ
				OutBitStream *dst_mcu = &dst[mcu_id];

				// 各画像ブロックの左上で有るかどうか(左上なら0,そうでないなら1)
				const u_int is_block_left_top = !((blockIdx.y == 0 || blockIdx.y == 4 || blockIdx.y == 5) && blockIdx.x == 0
					&& threadIdx.x == 0);

				// ----------------------------- DC成分 ------------------------------------
				// DC成分は前のMCUの成分との差分をハフマン符号化するため
				// 画像の最も左上のMCUとの差分は0と取る
				int diff = quantized[mcu_start_index]
					- is_block_left_top * quantized[mcu_start_index - 64 * is_block_left_top];

				// 差分の絶対値から最低限その値を表すのに必要なbitを計算
				// おそらく有効bitは11(=2^11=2048以下)のはず
				byte4 abs_diff = abs(diff);
				u_int effective_bits = EffectiveBits(abs_diff);
				dst_mcu->setBits(DCCodeTable[blockIdx.y / 2][effective_bits],
					DCSizeTable[blockIdx.y / 2][effective_bits]);

				// 0以外ならハフマンbitに続けて実値を書き込む
				if (effective_bits != 0) {
					diff -= (int) (diff < 0);
					dst_mcu->setBits(diff, effective_bits);
				}

				// ----------------------------- AC成分 ------------------------------------
				// 残り63のDCT係数に対して処理を行う
				u_int runlength = 0;

				// 末尾以外
				for (u_int i = 1; i < 63; i++) {
					int value = quantized[mcu_start_index + i];
					byte4 abs_value = abs(value);

					if (abs_value != 0) {
						// 0の個数が16ごとにZRL=code_id:151を代わりに割り当てる
						while (runlength > 15) {
							dst_mcu->setBits(ACCodeTable[blockIdx.y / 2][AC::luminance::ZRL],
								ACSizeTable[blockIdx.y / 2][AC::luminance::ZRL]);
							runlength -= 16;
						}

						// 有効bit数と15以下の0の個数を合わせて符号化コードを書き込む
						// おそらくAC成分の有効bitは10(=2^10=1024)以下のはず
						// したがってcode_idは[1,150][152,161]の範囲
						effective_bits = EffectiveBits(abs_value);
						u_int code_id = runlength * 10 + effective_bits + (runlength == 15);
						u_int size = ACSizeTable[blockIdx.y / 2][code_id];
						u_int code = ACCodeTable[blockIdx.y / 2][code_id];
						dst_mcu->setBits(code, size);
						runlength = 0;

						// ハフマンbitに続けて実値を書き込む
						value -= (int) (value < 0);
						dst_mcu->setBits(value, effective_bits);
					} else {
						++runlength;
					}
				}
				// 末尾はEOB=code_id:0
				dst_mcu->setBits(ACCodeTable[blockIdx.y / 2][AC::luminance::EOB],
					ACSizeTable[blockIdx.y / 2][AC::luminance::EOB]);
			}

			/**
			 * @brief ハフマンストリームの結合カーネル
			 *
			 * 排他処理のため3つに分ける.
			 * 1MCUは最小4bit(EOBのみ)なので1Byteのバッファに最大3MCUが競合するため.
			 * Scaleで分割数、Strideで各カーネルにおけるオフセットを指定
			 *
			 * - カーネル起動は8x8ブロックごと
			 *
			 * 	- grid(thread_devide, block_num, 1)
			 * 	- block(stream_size / block_num / Scale / thread_devide, 1, 1)
			 *
			 * @param stream 結合するストリーム配列へのポインタ
			 * @param info 結合情報配列へのポインタ
			 * @param dst 書き込み先
			 * @param block_size ブロックのサイズ
			 */
			template<u_int Scale, u_int Stride>
			__global__ void CombineHuffmanStream(const OutBitStream *stream,
				const OutBitStream::WriteBitsInfo *info, byte *dst, u_int block_size) {
				u_int mcu_id = (threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x) * Scale
					+ Stride;
				u_int block_id = blockIdx.y;

				OutBitStream::WriteStreamToLineBuffer(stream + mcu_id, info + mcu_id,
					dst + block_size * block_id);
			}
		} // namespace kernel

		//-------------------------------------------------------------------------------------------------//
		//
		// C/C++ CPU側インタフェース
		//
		//-------------------------------------------------------------------------------------------------//
		u_int CalcOptimumThreads(u_int require_threads) {
			u_int threads_per_block = require_threads;

			if (threads_per_block < 32) {
				return threads_per_block;
			}

			u_int result = 0;
			if (threads_per_block % 32 == 0) {
				result = 32;
			}
			if (threads_per_block % 64 == 0) {
				result = 64;
			}
			if (threads_per_block % 96 == 0) {
				result = 96;
			}
			if (threads_per_block % 128 == 0) {
				result = 128;
			}
			if (threads_per_block % 192 == 0) {
				result = 192;
			}
			if (threads_per_block % 256 == 0) {
				result = 256;
			}

			if (result != 0)
				return result;

			return gcd(512u, threads_per_block);
		}

		void HuffmanEncode(const DeviceIntBuffer &quantized, DeviceByteBuffer &result, IntBuffer &effective_bits) {
			using namespace kernel;

			typedef cuda_memory<OutBitStream> CudaBitStreams;
			typedef cuda_memory<OutBitStream::WriteBitsInfo> CudaStreamInfos;

			const u_int buffer_size = quantized.size();
			const u_int block_num = effective_bits.size();
			const u_int block_size = buffer_size / block_num;
			const u_int mcu_num = buffer_size / 64;

			// 各MCU用のバッファを作成
			CudaByteBuffer buffer(OutBitStream::MAX_BLOCK_SIZE * mcu_num);
			buffer.fill_zero();
			CudaBitStreams stream(mcu_num);
			for (u_int i = 0; i < mcu_num; ++i) {
				stream[i].setStreamBuffer(buffer.device_data() + OutBitStream::MAX_BLOCK_SIZE * i);
			}
			stream.sync_to_device();

			// 各MCU用ごとにエンコード
			dim3 block(CalcOptimumThreads(block_size / 6 / 64), 1, 1);
			dim3 grid(block_size / 6 / 64 / block.x, 6, block_num);
			HuffmanEncodeForMCU<<<grid,block>>>(quantized.device_data(), stream.device_data());

			// Test Code
			buffer.sync_to_host();
			CudaIntBuffer decode_qua_src(64);
			InBitStream ibs(buffer.host_data() + 128 * 2, 128);
			cpu::decode_huffman(&ibs, decode_qua_src.host_data(), 8, 8);
			DebugLog::dump_memory(buffer.host_data(), buffer.size(), "huffman.dat");

			// 書きこみ情報の作成
			stream.sync_to_host();
			CudaStreamInfos info(mcu_num);
			OutBitStream::CreateWriteBitsTable(stream.host_data(), info.host_data(), mcu_num, mcu_num / block_num);
			info.sync_to_device();

			// 各画像ブロックの有効bit数を算出
			for (u_int i = 0; i < block_num; ++i) {
				u_int last_index = (i + 1) * mcu_num / block_num - 1;
				effective_bits[i] = info[last_index].bits_of_grobal + info[last_index].bits_of_stream;
			}

			// 適切なスレッド数でMCUbitを結合
			u_int total_thread = mcu_num / block_num / 3;
			block = dim3(CalcOptimumThreads(total_thread), 1, 1);
			grid = dim3(total_thread / block.x, block_num, 1);

			CombineHuffmanStream<3, 0> <<<grid,block>>>(
				stream.device_data(), info.device_data(), result.device_data(), result.size() / block_num);
			CombineHuffmanStream<3, 1> <<<grid,block>>>(
				stream.device_data(), info.device_data(), result.device_data(), result.size() / block_num);
			CombineHuffmanStream<3, 2> <<<grid,block>>>(
				stream.device_data(), info.device_data(), result.device_data(), result.size() / block_num);
		}

		class Encoder::Impl {
		public:
			typedef cuda_memory<kernel::OutBitStream> CudaBitStreams;
			typedef cuda_memory<kernel::OutBitStream::WriteBitsInfo> CudaStreamInfos;

			u_int width_;
			u_int height_;
			u_int block_width_;
			u_int block_height_;

			u_int buffer_size_;
			u_int block_size_;
			u_int block_num_;
			u_int mcu_num_;

			u_int quarity_;

			DeviceTable conv_table_;
			DeviceByteBuffer yuv_result_;
			DeviceIntBuffer dct_result_;
			DeviceIntBuffer qua_result_;
			CudaByteBuffer huffman_result_;
			IntBuffer huffman_effective_bits_;

			DeviceByteBuffer mcu_buffer_;
			CudaBitStreams stream_;
			CudaStreamInfos info_;

		public:
			Impl(u_int width, u_int height, u_int block_width, u_int block_height) :
				width_(width),
				height_(height),
				block_width_(block_width),
				block_height_(block_height),
				buffer_size_(width * height * 3 / 2),
				block_size_(block_width * block_height * 3 / 2),
				block_num_(buffer_size_ / block_size_),
				mcu_num_(buffer_size_ / 64),
				quarity_(80),
				conv_table_(width * height),
				yuv_result_(buffer_size_),
				dct_result_(buffer_size_),
				qua_result_(buffer_size_),
				huffman_result_(buffer_size_),
				huffman_effective_bits_(block_num_),
				mcu_buffer_(kernel::OutBitStream::MAX_BLOCK_SIZE * mcu_num_),
				stream_(mcu_num_),
				info_(mcu_num_) {

				CreateConversionTable(width_, height_,
					block_width_, block_height_, conv_table_);

				mcu_buffer_.fill_zero();
				for (u_int i = 0; i < mcu_num_; ++i) {
					stream_[i].setStreamBuffer(
						mcu_buffer_.device_data() + kernel::OutBitStream::MAX_BLOCK_SIZE * i);
				}
				stream_.sync_to_device();
			}

			void reset() {
				CreateConversionTable(width_, height_,
					block_width_, block_height_, conv_table_);

				buffer_size_ = width_ * height_ * 3 / 2;
				block_size_ = block_width_ * block_height_ * 3 / 2;
				block_num_ = buffer_size_ / block_size_;
				mcu_num_ = buffer_size_ / 64;

				conv_table_.resize(width_ * height_, true);
				yuv_result_.resize(buffer_size_, true);
				dct_result_.resize(buffer_size_, true);
				qua_result_.resize(buffer_size_, true);
				huffman_result_.resize(buffer_size_, true);
				huffman_effective_bits_.resize(block_num_);

				mcu_buffer_.resize(kernel::OutBitStream::MAX_BLOCK_SIZE * mcu_num_, true);
				stream_.resize(mcu_num_, true);
				info_.resize(mcu_num_, true);

				mcu_buffer_.fill_zero();
				for (u_int i = 0; i < mcu_num_; ++i) {
					stream_[i].setStreamBuffer(
						mcu_buffer_.device_data() + kernel::OutBitStream::MAX_BLOCK_SIZE * i);
				}
				stream_.sync_to_device();
			}

			void setImageSize(u_int width, u_int height) {
				width_ = width;
				height_ = height;
			}

			void setBlockSize(u_int block_width, u_int block_height) {
				block_width_ = block_width;
				block_height_ = block_height;
			}

			void setQuarity(u_int quarity) {
				quarity_ = quarity;
			}

			void huffmanEncode() {
				// 各MCU用ごとにエンコード
				dim3 block(CalcOptimumThreads(block_size_ / 6 / 64), 1, 1);
				dim3 grid(block_size_ / 6 / 64 / block.x, 6, block_num_);
				kernel::HuffmanEncodeForMCU<<<grid,block>>>(
					qua_result_.device_data(), stream_.device_data());

				// 書きこみ情報の作成
				stream_.sync_to_host();
				kernel::OutBitStream::CreateWriteBitsTable(
					stream_.host_data(), info_.host_data(), mcu_num_, mcu_num_ / block_num_);
				info_.sync_to_device();

				// 各画像ブロックの有効bit数を算出
				for (u_int i = 0; i < block_num_; ++i) {
					u_int last_index = (i + 1) * mcu_num_ / block_num_ - 1;
					huffman_effective_bits_[i] = info_[last_index].bits_of_grobal + info_[last_index].bits_of_stream;
				}

				// 適切なスレッド数でMCUbitを結合
				u_int total_thread = mcu_num_ / block_num_ / 3;
				block = dim3(CalcOptimumThreads(total_thread), 1, 1);
				grid = dim3(total_thread / block.x, block_num_, 1);

				kernel::CombineHuffmanStream<3, 0><<<grid,block>>>(
					stream_.device_data(), info_.device_data(), huffman_result_.device_data(), huffman_result_.size() / block_num_);
				kernel::CombineHuffmanStream<3, 1><<<grid,block>>>(
					stream_.device_data(), info_.device_data(), huffman_result_.device_data(), huffman_result_.size() / block_num_);
				kernel::CombineHuffmanStream<3, 2><<<grid,block>>>(
					stream_.device_data(), info_.device_data(), huffman_result_.device_data(), huffman_result_.size() / block_num_);

				huffman_result_.sync_to_host();
			}

			void encode(const DeviceByteBuffer &rgb) {
				ConvertRGBToYUV(rgb,
					yuv_result_,
					width_,
					height_,
					block_width_,
					block_height_,
					conv_table_);
				DiscreteCosineTransform(
					yuv_result_,
					dct_result_);
				ZigzagQuantize(
					dct_result_,
					qua_result_,
					block_width_ * block_height_ * 3 / 2,
					quarity_);
				huffmanEncode();
			}

			Encoder::Result getEncodedData(u_int block_index) {
				Encoder::Result r = {
					huffman_result_.host_data() + block_index * huffman_result_.size() / block_num_,
					huffman_effective_bits_[block_index]
				};
				return r;
			}
		};

		Encoder::Encoder(u_int width, u_int height) :
			impl(new Impl(width, height, width, height)) {

		}

		Encoder::Encoder(u_int width, u_int height, u_int block_width, u_int block_height) :
			impl(new Impl(width, height, block_width, block_height)) {

		}

		Encoder::~Encoder() {
			delete impl;
		}

		void Encoder::reset() {
			impl->reset();
		}

		void Encoder::setQuarity(u_int quarity) {
			impl->setQuarity(quarity);
		}

		void Encoder::setImageSize(u_int width, u_int height) {
			impl->setImageSize(width, height);
		}

		void Encoder::setBlockSize(u_int block_width, u_int block_height) {
			impl->setBlockSize(block_width, block_height);
		}

		void Encoder::encode(const DeviceByteBuffer &rgb) {
			impl->encode(rgb);
		}
	} // namespace cuda
} // namespace jpeg
