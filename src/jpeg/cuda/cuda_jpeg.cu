#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <jpeg/cpu/cpu_jpeg.h>
#include <jpeg/cuda/cuda_jpeg.cuh>

#include <utils/debug_log.h>
#include <utils/type_definitions.h>
#include <utils/utility.hpp>
#include <utils/cuda/bit_operation.cuh>
#include <utils/cuda/cuda_timer.h>

#include "encoder_tables_device.cuh"

namespace jpeg {
	namespace cuda {

		using namespace util;
		using namespace util::cuda;
		using namespace encode_table;

		using util::u_int;

		//-------------------------------------------------------------------------------------------------
		//
		// debugコピペ用
		//
		// printf("%d\n", gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z);
		// printf("%d, %d, %d, %d, %d, %d\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
		// printf("%d, %d, %d, %d, %d, %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
		// printf("src, %d, dst, %d\n", src, dst);
		//
		//-------------------------------------------------------------------------------------------------

		/**
		 * CUDAカーネル関数
		 */
		namespace kernel {
			//-------------------------------------------------------------------------------------------------//
			//
			// 色変換
			//
			//-------------------------------------------------------------------------------------------------//
			/**
			 * @brief 色変換テーブル作成カーネル
			 *
			 * - カーネル起動はピクセル数
			 * - カーネル起動は必ず次のブロック/グリッドで行われなければならない
			 * 	- grid(block_width/16, block_height/16, width/block_width * height/block_height)
			 * 	- block(16, 16, 1)
			 *
			 * TODO 変換はできてる。全体なら問題ない。部分抜き出しでなぜ違うのか要検討。
			 *
			 * @param width もと画像の幅
			 * @param height 元画像の高さ
			 * @param block_width ブロックの幅
			 * @param block_height ブロックの高さ
			 * @param table テーブル出力
			 */
			void __global__ CreateConversionTable(u_int width, u_int height, u_int block_width,
				u_int block_height, TableElementSrcToDst *table) {

				const u_int img_block_x_num = width / block_width;
				const u_int img_block_size = block_width * block_height;
				const u_int mcu_block_x_num = block_width / 16;

				const u_int block_x = blockIdx.z % img_block_x_num;
				const u_int block_y = blockIdx.z / img_block_x_num;
				const u_int block_id = block_x + block_y * img_block_x_num;
				const u_int src_block_start_index = block_y * width * block_height + block_x * block_width;
				const u_int dst_block_start_y_index = block_id * img_block_size * 3 / 2;
				const u_int dst_block_start_u_index = dst_block_start_y_index + img_block_size;
				const u_int dst_block_start_v_index = dst_block_start_u_index + img_block_size / 4;

				const u_int mcu_x = blockIdx.x;
				const u_int mcu_y = blockIdx.y;
				const u_int mcu_id = mcu_x + mcu_y * mcu_block_x_num;
				const u_int src_mcu_start_index = src_block_start_index + mcu_y * width * 16 + mcu_x * 16;
				const u_int dst_mcu_y_start_index = dst_block_start_y_index + mcu_id * 256;
				const u_int dst_mcu_u_start_index = dst_block_start_u_index + mcu_id * 64;
				const u_int dst_mcu_v_start_index = dst_block_start_v_index + mcu_id * 64;

				const u_int pix_x = threadIdx.x;
				const u_int pix_y = threadIdx.y;

				const u_int mcu_id_x = pix_x / 8; // 0,1
				const u_int mcu_id_y = pix_y / 8; // 0,1
				const u_int block_8x8_id = mcu_id_x + 2 * mcu_id_y; // 0-3
				const u_int dst_mcu_y_8x8_index = pix_x % 8 + (pix_y % 8) * 8; // 0-63
				const u_int x = pix_x / 2, y = pix_y / 2; // 0-63

				// RGB画像のピクセルインデックス
				const u_int src_index = src_mcu_start_index + pix_x + pix_y * width;
				// YUVの書き込みインデックス
				const u_int dst_y_index = dst_mcu_y_start_index + block_8x8_id * 64 + dst_mcu_y_8x8_index;
				const u_int dst_u_index = dst_mcu_u_start_index + x + y * 8;
				const u_int dst_v_index = dst_mcu_v_start_index + x + y * 8;

				table[src_index].y = dst_y_index;
				table[src_index].u = dst_u_index;
				table[src_index].v = dst_v_index;
			}

			/**
			 * @brief RGB→YUV変換カーネル
			 *
			 * - カーネル起動は各ピクセルごと = width*heightスレッド必要
			 * 	- grid(block_width/16, block_height/16, width/block_width * height/block_height)
			 * 	- block(16, 16, 1)
			 * - グリッド/ブロック数に制限はない
			 * - rgb.size == yuv.sizeであること
			 *
			 * TODO おそらくバス幅を考えると32bit単位でのアクセスが最適.
			 * TODO すなわち例えば3byte*4要素=12byteごとに手動unrollしたほうが
			 * TODO 早くなるかもしれない.逆変換も同様に.
			 *
			 * @param rgb BGRで保存されたソースデータ
			 * @param yuv_result yuvに変換された結果
			 * @param table 変換テーブル
			 */
			void __global__ ConvertRGBToYUV(const byte* rgb, byte* yuv_result,
				const TableElementSrcToDst *table) {
				const u_int pix_index = threadIdx.x + threadIdx.y * blockDim.x
					+ (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * blockDim.x
						* blockDim.y;

				const TableElementSrcToDst elem = table[pix_index];
				const u_int src_index = pix_index * 3;

				// R,G,B [0, 255] -> R,G,B [16, 235]
				const float b = rgb[src_index + 0];
				const float g = rgb[src_index + 1];
				const float r = rgb[src_index + 2];

				// R,G,B [16, 235] -> Y [16, 235] U,V [16, 240]
				yuv_result[elem.y] = 0.257 * r + 0.504 * g + 0.098 * b + 16;
				yuv_result[elem.u] = -0.148 * r - 0.291 * g + 0.439 * b + 128;
				yuv_result[elem.v] = 0.439 * r - 0.368 * g - 0.071 * b + 128;
			}

			__device__ byte revise_value_d(double v) {
				if (v < 0.0)
					return 0;
				if (v > 255.0)
					return 255;
				return (byte) v;
			}
			/**
			 * @brief YUV→RGB変換カーネル
			 *
			 * - カーネル起動は各ピクセルごと = width*heightスレッド必要
			 * 	- grid(block_width/16, block_height/16, width/block_width * height/block_height)
			 * 	- block(16, 16, 1)
			 * - グリッド/ブロック数に制限はない
			 * - rgb.size == yuv.sizeであること
			 *
			 * @param yuv YUV411で保存されたソースデータ
			 * @param rgb_result rgbに変換された結果
			 * @param table 変換テーブル
			 */
			void __global__ ConvertYUVToRGB(const byte* yuv, byte* rgb_result,
				const TableElementSrcToDst *table) {
				const u_int pix_index = threadIdx.x + threadIdx.y * blockDim.x
					+ (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * blockDim.x
						* blockDim.y;

				const TableElementSrcToDst elem = table[pix_index];
				const u_int dst_index = pix_index * 3;

				// Y [16, 235] U,V [16, 240] -> Y [16, 235] U,V [-112, 112]
				const float y = yuv[elem.y] - 16;
				const float u = yuv[elem.u] - 128.0f;
				const float v = yuv[elem.v] - 128.0f;

				// Y [16, 235] U,V [-112, 112] -> R,G,B [0, 255]
				rgb_result[dst_index + 0] = 1.164 * y + 2.018 * u;
				rgb_result[dst_index + 1] = 1.164 * y - 0.391 * u - 0.813 * v;
				rgb_result[dst_index + 2] = 1.164 * y + 1.596 * v;
			}

			//-------------------------------------------------------------------------------------------------//
			//
			// DCT/iDCT
			//
			//-------------------------------------------------------------------------------------------------//
			/**
			 * @brief DCTカーネル
			 *
			 * - カーネル起動は各YUVごと = width*height*3/2スレッド必要
			 * 	- grid(yuv.size() / 64 / 6, 1, 1)
			 * 	- block(8, 8, 6)
			 * - グリッド/ブロック数に制限はない
			 * - dst_coefficient.size == yuv.sizeであること
			 *
			 * @param yuv_src 64byte=8x8blockごとに連続したメモリに保存されたデータ
			 * @param dct_coefficient DCT係数
			 */
			void __global__ DiscreteCosineTransform(const byte *yuv_src, int *dct_coefficient) {
				using DCTConstants::cos;
				using DCTConstants::cos_t;

				u_int x = threadIdx.x, y = threadIdx.y;
				u_int local_index = x + y * 8;
				u_int start_index = 384 * blockIdx.x + 64 * threadIdx.z;

				__shared__ float vertical_result[6][64];

				vertical_result[threadIdx.z][local_index] = cos[y * 8 + 0] * yuv_src[start_index + x + 0 * 8]
					+ cos[y * 8 + 1] * yuv_src[start_index + x + 1 * 8]
					+ cos[y * 8 + 2] * yuv_src[start_index + x + 2 * 8]
					+ cos[y * 8 + 3] * yuv_src[start_index + x + 3 * 8]
					+ cos[y * 8 + 4] * yuv_src[start_index + x + 4 * 8]
					+ cos[y * 8 + 5] * yuv_src[start_index + x + 5 * 8]
					+ cos[y * 8 + 6] * yuv_src[start_index + x + 6 * 8]
					+ cos[y * 8 + 7] * yuv_src[start_index + x + 7 * 8];

				__syncthreads();

				dct_coefficient[start_index + local_index] = vertical_result[threadIdx.z][y * 8 + 0] * cos_t[x + 0 * 8]
					+ vertical_result[threadIdx.z][y * 8 + 1] * cos_t[x + 1 * 8]
					+ vertical_result[threadIdx.z][y * 8 + 2] * cos_t[x + 2 * 8]
					+ vertical_result[threadIdx.z][y * 8 + 3] * cos_t[x + 3 * 8]
					+ vertical_result[threadIdx.z][y * 8 + 4] * cos_t[x + 4 * 8]
					+ vertical_result[threadIdx.z][y * 8 + 5] * cos_t[x + 5 * 8]
					+ vertical_result[threadIdx.z][y * 8 + 6] * cos_t[x + 6 * 8]
					+ vertical_result[threadIdx.z][y * 8 + 7] * cos_t[x + 7 * 8];
			}

			/**
			 * @brief iDCTカーネル
			 *
			 * - カーネル起動は各YUVごと = width*height*3/2スレッド必要
			 * 	- grid(yuv.size() / 64 / 6, 1, 1)
			 * 	- block(8, 8, 6)
			 * - グリッド/ブロック数に制限はない
			 * - dst_coefficient.size == yuv.sizeであること
			 *
			 * @param dct_coefficient DCT係数
			 * @param yuv_result 64byte=8x8blockごとに連続したメモリに保存されたデータ
			 */
			void __global__ InverseDiscreteCosineTransform(const int *dct_coefficient, byte *yuv_result) {
				using DCTConstants::cos;
				using DCTConstants::cos_t;

				u_int x = threadIdx.x, y = threadIdx.y;
				u_int local_index = x + y * 8;
				u_int start_index = 384 * blockIdx.x + 64 * threadIdx.z;

				__shared__ float vertical_result[6][64];

				vertical_result[threadIdx.z][local_index] = cos_t[y * 8 + 0] * dct_coefficient[start_index + x + 0 * 8]
					+ cos_t[y * 8 + 1] * dct_coefficient[start_index + x + 1 * 8]
					+ cos_t[y * 8 + 2] * dct_coefficient[start_index + x + 2 * 8]
					+ cos_t[y * 8 + 3] * dct_coefficient[start_index + x + 3 * 8]
					+ cos_t[y * 8 + 4] * dct_coefficient[start_index + x + 4 * 8]
					+ cos_t[y * 8 + 5] * dct_coefficient[start_index + x + 5 * 8]
					+ cos_t[y * 8 + 6] * dct_coefficient[start_index + x + 6 * 8]
					+ cos_t[y * 8 + 7] * dct_coefficient[start_index + x + 7 * 8];

				__syncthreads();

				float value = vertical_result[threadIdx.z][y * 8 + 0] * cos[x + 0 * 8]
					+ vertical_result[threadIdx.z][y * 8 + 1] * cos[x + 1 * 8]
					+ vertical_result[threadIdx.z][y * 8 + 2] * cos[x + 2 * 8]
					+ vertical_result[threadIdx.z][y * 8 + 3] * cos[x + 3 * 8]
					+ vertical_result[threadIdx.z][y * 8 + 4] * cos[x + 4 * 8]
					+ vertical_result[threadIdx.z][y * 8 + 5] * cos[x + 5 * 8]
					+ vertical_result[threadIdx.z][y * 8 + 6] * cos[x + 6 * 8]
					+ vertical_result[threadIdx.z][y * 8 + 7] * cos[x + 7 * 8];

				yuv_result[start_index + local_index] = (byte) ((int) value);
			}

			//-------------------------------------------------------------------------------------------------//
			//
			// 量子化
			//
			//-------------------------------------------------------------------------------------------------//
			/**
			 * @brief 量子化用のテーブル
			 */
			static __device__ __constant__ const int* quantize_table[] = {
				Quantize::luminance, Quantize::luminance, Quantize::component };

			/**
			 * @brief 低品質ジグザグ量子化カーネル
			 *
			 * - カーネル起動は各DCT係数ごと = width*height*3/2スレッド必要
			 * 	- grid([[block_size / 3] / 128], [3:table switch], [block_num])
			 * 	- block(8, 8, 2)
			 *
			 * @param dct_coefficient DCT係数行列
			 * @param quantized 量子化データ
			 * @param quality 量子化品質[0,100]
			 */
			void __global__ ZigzagQuantizeLow(const int *dct_coefficient, int *quantized, float quality) {
				using Zigzag::sequence;

				u_int local_index = threadIdx.x + threadIdx.y * 8;
				u_int start_index = 64 * threadIdx.z + 128 * blockIdx.x + 128 * gridDim.x * blockIdx.y
					+ gridDim.x * 128 * 3 * blockIdx.z;
				quantized[start_index + sequence[local_index]] = dct_coefficient[start_index + local_index]
					/ ((255.0f - ((255.0f - quantize_table[blockIdx.y][local_index]) * (1.0f + quality))));
			}

			/**
			 * @brief 低品質逆ジグザグ量子化カーネル
			 *
			 * - カーネル起動は各DCT係数ごと = width*height*3/2スレッド必要
			 * 	- grid([[block_size / 3] / 128], [3:table switch], [block_num])
			 * 	- block(8, 8, 2)
			 *
			 * @param quantized 量子化データ
			 * @param dct_coefficient DCT係数行列
			 * @param quality 量子化品質[0,100]
			 */
			void __global__ InverseZigzagQuantizeLow(const int *quantized, int *dct_coefficient,
				float quality) {
				using Zigzag::sequence;

				u_int local_index = threadIdx.x + threadIdx.y * 8;
				u_int start_index = 64 * threadIdx.z + 128 * blockIdx.x + 128 * gridDim.x * blockIdx.y
					+ gridDim.x * 128 * 3 * blockIdx.z;
				dct_coefficient[start_index + local_index] = quantized[start_index + sequence[local_index]]
					* ((255.0f - ((255.0f - quantize_table[blockIdx.y][local_index]) * (1.0f + quality))));
			}

			/**
			 * @brief 高品質ジグザグ量子化カーネル
			 *
			 * - カーネル起動は各DCT係数ごと = width*height*3/2スレッド必要
			 * 	- grid([[block_size / 3] / 128], [3:table switch], [block_num])
			 * 	- block(8, 8, 2)
			 *
			 * @param dct_coefficient DCT係数行列
			 * @param quantized 量子化データ
			 * @param quality 量子化品質[0,100]
			 */
			void __global__ ZigzagQuantizeHigh(const int *dct_coefficient, int *quantized, float quality) {
				using Zigzag::sequence;

				u_int local_index = threadIdx.x + threadIdx.y * 8;
				u_int start_index = 64 * threadIdx.z + 128 * blockIdx.x + 128 * gridDim.x * blockIdx.y
					+ gridDim.x * 128 * 3 * blockIdx.z;
				quantized[start_index + sequence[local_index]] = dct_coefficient[start_index + local_index]
					/ (quantize_table[blockIdx.y][local_index] * (1.0f - quality));
			}

			/**
			 * @brief 高品質逆ジグザグ量子化カーネル
			 *
			 * - カーネル起動は各DCT係数ごと = width*height*3/2スレッド必要
			 * 	- grid([[block_size / 3] / 128], [3:table switch], [block_num])
			 * 	- block(8, 8, 2)
			 *
			 * @param quantized 量子化データ
			 * @param dct_coefficient DCT係数行列
			 * @param quality 量子化品質[0,100]
			 */
			void __global__ InverseZigzagQuantizeHigh(const int *quantized, int *dct_coefficient,
				float quality) {
				using Zigzag::sequence;

				u_int local_index = threadIdx.x + threadIdx.y * 8;
				u_int start_index = 64 * threadIdx.z + 128 * blockIdx.x + 128 * gridDim.x * blockIdx.y
					+ gridDim.x * 128 * 3 * blockIdx.z;
				dct_coefficient[start_index + local_index] = quantized[start_index + sequence[local_index]]
					* (quantize_table[blockIdx.y][local_index] * (1.0f - quality));
			}

			/**
			 * @brief 最高品質(準無劣化)ジグザグ量子化カーネル
			 *
			 * 最高品質の場合量子化テーブルを用いないためにYUVを区別する必要はない.
			 * また、MCUは16x16であることから、最低要素数は16x16x3/2=384である.
			 * メモリアクセスの観点からおそらく192 thread/blockがいいかも.
			 *
			 * - カーネル起動は各DCT係数ごと = width*height*3/2スレッド必要
			 * 	- grid([[block_size / 3] / 128], [3:table switch], [block_num])
			 * 	- block(8, 8, 2)
			 *
			 * @param dct_coefficient DCT係数行列
			 * @param quantized 量子化データ
			 */
			void __global__ ZigzagQuantizeMax(const int *dct_coefficient, int *quantized) {
				using Zigzag::sequence;

				u_int local_index = threadIdx.x + threadIdx.y * 8;
				u_int start_index = 64 * threadIdx.z + 192 * blockIdx.x;
				quantized[start_index + sequence[local_index]] = dct_coefficient[start_index + local_index];
			}

			/**
			 * @brief 最高品質(準無劣化)逆ジグザグ量子化カーネル
			 *
			 * 最高品質の場合量子化テーブルを用いないためにYUVを区別する必要はない.
			 * また、MCUは16x16であることから、最低要素数は16x16x3/2=384である.
			 * メモリアクセスの観点からおそらく192 thread/blockがいいかも.
			 *
			 * - カーネル起動は各DCT係数ごと = width*height*3/2スレッド必要
			 * 	- grid([[block_size / 3] / 128], [3:table switch], [block_num])
			 * 	- block(8, 8, 2)
			 *
			 * @param quantized 量子化データ
			 * @param dct_coefficient DCT係数行列
			 */
			void __global__ InverseZigzagQuantizeMax(const int *quantized, int *dct_coefficient) {
				using Zigzag::sequence;

				u_int local_index = threadIdx.x + threadIdx.y * 8;
				u_int start_index = 64 * threadIdx.z + 192 * blockIdx.x;
				dct_coefficient[start_index + local_index] = quantized[start_index + sequence[local_index]];
			}

			using namespace cuda::encode_table::HuffmanEncode;

			/** @brief 余ったビットに1を詰めるためのマスク */
			static __device__ __constant__ const unsigned char kBitFullMaskT[] = {
				0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0x7f, 0xff };
			/** @brief 余ったビットに1を詰めるためのマスク */
			static __device__ __constant__ const unsigned char kBitFullMaskLowT[] = {
				0xff, 0x7f, 0x3f, 0x1f, 0x0f, 0x07, 0x03, 0x01 };
			// ビット取り出しのためのマスク
			static __device__ __constant__ const unsigned char kBitTestMaskT[8] = {
				0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 };

			//-------------------------------------------------------------------------------------------------//
			//
			// ハフマン符号化
			//
			//-------------------------------------------------------------------------------------------------//
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
				 * @brief リセットする
				 */
				inline __device__ __host__ void reset() {
					byte_pos_ = 0;
					bit_pos_ = 7;
					writable_ = 1;
				}

				/**
				 * @brief リセットする
				 *
				 * @param device_buf デバイスメモリへのポインタ
				 */
				inline __device__ __host__ void reset(byte* device_buf) {
					reset();
					dst_buf_ = device_buf;
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
				static __host__ void CreateWriteBitsTable(const OutBitStream *stream, WriteBitsInfo* num_bits,
					u_int stream_num, u_int stream_per_block) {

					u_int blocks = stream_num / stream_per_block;

					for (u_int block_index = 0; block_index < blocks; ++block_index) {
						//BitPosは7が上位で0が下位なので注意,更に位置なので注意。7なら要素は0,0なら要素は7
						u_int first_stream_of_block = block_index * stream_per_block;

						num_bits[first_stream_of_block].bits_of_stream = stream[first_stream_of_block]
							.byte_pos_ * 8 + (7 - stream[first_stream_of_block].bit_pos_);
						num_bits[first_stream_of_block].bits_of_grobal = 0;

						for (u_int j = first_stream_of_block + 1;
							j < first_stream_of_block + stream_per_block; ++j) {
							num_bits[j].bits_of_stream = stream[j].byte_pos_ * 8 + (7 - stream[j].bit_pos_);
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
				dst_mcu->reset();

				// 各画像ブロックの左上で有るかどうか(左上なら0,そうでないなら1)
				const u_int is_block_left_top = !((blockIdx.y == 0 || blockIdx.y == 4 || blockIdx.y == 5)
					&& blockIdx.x == 0 && threadIdx.x == 0);

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
#pragma unroll
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

//			/**
//			 * @brief ビット読み出しクラス
//			 *
//			 * コンストラクタに渡した読み込みバッファは、
//			 * このクラスの寿命より先に破棄されてはいけない
//			 *
//			 * @author AsWe.Co modified by momma
//			 * @version 1.0
//			 */
//			class InBitStream {
//			public:
//				/**
//				 * @brief コンストラクタ
//				 *
//				 * @param aBufP 読み込みバッファ
//				 * @param size バッファの有効サイズ
//				 */
//				__device__ InBitStream(const byte* aBufP, size_t size) {
//					// データバッファの設定
//					mBufP = (byte*) aBufP; // バッファ
//					mEndOfBufP = mBufP + size; // バッファの最終アドレス
//
//					// 状態変数初期化
//					mBitPos = 7; // 最上位ビット
//					mNextFlag = 1; // 読み飛ばし無し
//					mReadFlag = 1; // アクセスエラー無し
//				}
//
//				/**
//				 * @brief ビット単位で読み出す
//				 *
//				 * @param numOfBits 読みだすビット数
//				 * @return 読み出し値
//				 */
//				__device__ int getBits(size_t numOfBits) {
//					if (numOfBits <= 0)
//						return 0; // エラー
//
//					int r = 0; // 返値
//					while (numOfBits) {
//						if (mBitPos < 0) { // 次のバイトを読み出すとき
//							mBitPos = 7; // 読み出しビット位置更新
//							incBuf(); // アドレス更新
//						}
//
//						// 返値の作成
//						r <<= 1;
//						r |= ((*mBufP) & kBitTestMaskT[mBitPos--]) ? 1 : 0;
//						// 1ビット読み出しと読み出しビット位置更新
//						numOfBits--; // 読み出しビット数更新
//					}
//					return r;
//				}
//
//			private:
//				byte* mBufP; 		//! 読み出しアドレス
//				byte* mEndOfBufP; 	//! バッファの終了アドレス
//				int mBitPos; 		//! 読み出しビット位置（上位ビットが7、下位ビットが0）
//				int mNextFlag; 		//! 次のバイトを読んでいいかどうか
//				int mReadFlag; 		//! 1:読み出し可, 0:読み出し不可
//
//				/** @brief 読み出しアドレスのインクリメントとアクセス違反のチェック */
//				__device__ void incBuf() {
//					if (++mBufP >= mEndOfBufP) // 次のアクセスでエラー
//						mReadFlag = 0;
//				}
//
//				__device__ InBitStream(jpeg::cuda::kernel::InBitStream &);
//				void __device__ operator =(jpeg::cuda::kernel::InBitStream &);
//			};
//
//			__device__ int decode_huffman_word(jpeg::cuda::kernel::InBitStream *ibit_stream, int is_ac,
//				int is_lumi) { // is_lumi =0で輝度値ってことにする
//				// ハフマンテーブル指定
//				using HuffmanDecode::TableSet;
//				using HuffmanDecode::tableSet;
//				const TableSet &theHT = tableSet[is_lumi][is_ac];	// 使用するハフマンテーブル
//
//				int code = 0; // ハフマン符号語の候補：最大値16ビット
//				int length = 0; // ハフマン符号語候補のビット数
//				int next = 0; // 次の1ビット
//				int k = 0; // 表の指数
//
//				while (k < theHT.table_size && length < 16) {
//					length++;
//					code <<= 1;
//					next = ibit_stream->getBits(1);
//
//					code |= next;
//
//					while (theHT.size_table[k] == length) { // 候補と符号語のビット数が等しい間検索
//						if (theHT.code_table[k] == code) { // ヒット
//							return theHT.value_table[k]; // 復号結果を返す
//						}
//						k++; // 次の符号語
//					}
//				}
//				return 0;
//			}
//
//			// grid : all_blocks/8x8blocks/3, 3, 1
//			// block : 8x8blocks, 1, 1
//			__global__ void HuffmanDecodeForMCU(const byte *huffman, int *offsetbits_of_block, int *dst_qua) {
//				using namespace encode_table::HuffmanEncode;
//				using jpeg::cuda::kernel::InBitStream;
//
//				// マクロブロック番号
//				const u_int mcu_id = threadIdx.x + gridDim.x * blockDim.x;
//				InBitStream ibit_stream(huffman + (u_int) ceil(offsetbits_of_block[mcu_id] / 8.0f), 255);
//				ibit_stream.getBits(offsetbits_of_block[mcu_id] % 8);
//				int is_lumi = blockIdx.y / 2;
//
//				int preDC = 0;
//
//				//--------------------------- DC ---------------------------
//				int diff = 0;
//				int category = decode_huffman_word(&ibit_stream, 0, is_lumi);
//
//				diff = ibit_stream.getBits(category);
//				if ((diff & (1 << (category - 1))) == 0) { //負
//					diff -= (1 << category) - 1;
//				}
//
//				preDC += diff;
//				dst_qua[mcu_id * 64] = preDC;
//
//				//--------------------------- AC ---------------------------
//				int k = 1;
//				while (k < 64) {
//					category = decode_huffman_word(&ibit_stream, 1, is_lumi);
//					if (category == 0) { //EOB
//						while (k < 64) {
//							dst_qua[mcu_id * 64 + (k++)] = 0;
//						}
//						break;
//					}
//
//					int run = category >> 4; //run length
//					category &= 0x0f; //category
//					int acv = 0;
//					if (category) {
//						acv = ibit_stream.getBits(category);
//						if ((acv & (1 << (category - 1))) == 0)
//							acv -= (1 << category) - 1; //負
//					}
//
//					while (run-- > 0) { //ランレングスの数だけ0
//						dst_qua[mcu_id * 64 + (k++)] = 0;
//					}
//					dst_qua[mcu_id * 64 + (k++)] = acv;
//				}
//			}

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
				u_int mcu_id = (threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x)
					* Scale + Stride;
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
		void CreateConversionTable(size_t width, size_t height, size_t block_width, size_t block_height,
			DeviceTable &table) {
			assert(table.size() >= width * height);
			const dim3 grid(block_width / 16, block_height / 16, width / block_width * height / block_height);
			const dim3 block(16, 16, 1);

			kernel::CreateConversionTable<<<grid, block>>>(width, height, block_width, block_height,
				table.device_data());
		}

		void ConvertRGBToYUV(const DeviceByteBuffer &rgb, DeviceByteBuffer &yuv_result, size_t width,
			size_t height, size_t block_width, size_t block_height, const DeviceTable &table) {
			assert(rgb.size()/2 <= yuv_result.size());

			const dim3 grid(block_width / 16, block_height / 16, width / block_width * height / block_height);
			const dim3 block(16, 16, 1);

			kernel::ConvertRGBToYUV<<<grid, block>>>(rgb.device_data(), yuv_result.device_data(),
				table.device_data());
		}

		void ConvertYUVToRGB(const DeviceByteBuffer &yuv, DeviceByteBuffer &rgb_result, size_t width,
			size_t height, size_t block_width, size_t block_height, const DeviceTable &table) {
			assert(yuv.size() <= rgb_result.size()/2);

			const dim3 grid(block_width / 16, block_height / 16, width / block_width * height / block_height);
			const dim3 block(16, 16, 1);

			kernel::ConvertYUVToRGB<<<grid, block>>>(yuv.device_data(), rgb_result.device_data(),
				table.device_data());
		}

		void DiscreteCosineTransform(const DeviceByteBuffer &yuv, DeviceIntBuffer &dct_coefficient) {
			// grid (8x8ブロックの個数, 分割数, 1)
			const dim3 grid(yuv.size() / 64 / 6, 1, 1);
			// grid (1, 8x8ブロック)
			const dim3 block(8, 8, 6);

			kernel::DiscreteCosineTransform<<<grid, block>>>(yuv.device_data(),
				dct_coefficient.device_data());
		}

		void InverseDiscreteCosineTransform(const DeviceIntBuffer &dct_coefficient,
			DeviceByteBuffer &yuv_result) {
			// grid (8x8ブロックの個数, 分割数, 1)
			const dim3 grid(dct_coefficient.size() / 64 / 6, 1, 1);
			// grid (1, 8x8ブロック)
			const dim3 block(8, 8, 6);

			kernel::InverseDiscreteCosineTransform<<<grid, block>>>(dct_coefficient.device_data(),
				yuv_result.device_data());
		}

		void CalculateDCTMatrix(float *dct_mat) {
			const float PI = 3.141592653589793f;
			for (u_int k = 0; k < 8; ++k) {
				for (u_int n = 0; n < 8; ++n) {
					float c = (k == 0) ? 1.0f / 1.41421356f : 1.0f;
					u_int index = n + k * 8;
					dct_mat[index] = c * 0.5f * cos((2.0f * n + 1) * k * PI / 16.0f);
				}
			}
		}

		void CalculateiDCTMatrix(float *idct_mat) {
			const float PI = 3.141592653589793f;
			for (u_int n = 0; n < 8; ++n) {
				for (u_int k = 0; k < 8; ++k) {
					float c = (n == 0) ? 1.0f / 1.41421356f : 1.0f;
					u_int index = n + k * 8;
					idct_mat[index] = c * 0.5f * cos((2 * k + 1) * n * PI / 16.0f);
				}
			}
		}

		void ZigzagQuantize(const DeviceIntBuffer &dct_coefficient, DeviceIntBuffer &quantized,
			u_int block_size, u_int quality) {
			// 最低品質
			if (quality == 0) {
				const dim3 grid(quantized.size() / 64 / 3, 3, 1);
				const dim3 block(8, 8, 1);
				kernel::ZigzagQuantizeLow<<<grid, block>>>(dct_coefficient.device_data(),
					quantized.device_data(), -1.0f);
			}
			// 低品質
			else if (quality < 50) {
				const dim3 grid(block_size / 128 / 3, 3, dct_coefficient.size() / block_size);
				const dim3 block(8, 8, 2);
				kernel::ZigzagQuantizeLow<<<grid, block>>>(dct_coefficient.device_data(),
					quantized.device_data(), (quality - 50.0f) / 50.0f);
			}
			// 高品質
			else if (quality < 100) {
				const dim3 grid(block_size / 128 / 3, 3, dct_coefficient.size() / block_size);
				const dim3 block(8, 8, 2);
				kernel::ZigzagQuantizeHigh<<<grid, block>>>(dct_coefficient.device_data(),
					quantized.device_data(), (quality - 50.0f) / 50.0f);
			}
			// 最高品質
			else {
				const dim3 grid(quantized.size() / 192, 3, 1);
				const dim3 block(8, 8, 3);
				kernel::ZigzagQuantizeMax<<<grid, block>>>(dct_coefficient.device_data(),
					quantized.device_data());
			}
		}

		void InverseZigzagQuantize(const DeviceIntBuffer &quantized, DeviceIntBuffer &dct_coefficient,
			u_int block_size, u_int quality) {
			// 最低品質
			if (quality == 0) {
				const dim3 grid(quantized.size() / 64 / 3, 3, 1);
				const dim3 block(8, 8, 1);
				kernel::InverseZigzagQuantizeLow<<<grid, block>>>(quantized.device_data(),
					dct_coefficient.device_data(), -1.0f);
			}
			// 低品質
			else if (quality < 50) {
				const dim3 grid(block_size / 128 / 3, 3, dct_coefficient.size() / block_size);
				const dim3 block(8, 8, 2);
				kernel::InverseZigzagQuantizeLow<<<grid, block>>>(quantized.device_data(),
					dct_coefficient.device_data(), (quality - 50.0f) / 50.0f);
			}
			// 高品質
			else if (quality < 100) {
				const dim3 grid(block_size / 128 / 3, 3, dct_coefficient.size() / block_size);
				const dim3 block(8, 8, 2);
				kernel::InverseZigzagQuantizeHigh<<<grid, block>>>(quantized.device_data(),
					dct_coefficient.device_data(), (quality - 50.0f) / 50.0f);
			}
			// 最高品質
			else {
				const dim3 grid(quantized.size() / 192, 1, 1);
				const dim3 block(8, 8, 3);
				kernel::InverseZigzagQuantizeMax<<<grid, block>>>(quantized.device_data(),
					dct_coefficient.device_data());
			}
		}

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

		void HuffmanEncode(const DeviceIntBuffer &quantized, DeviceByteBuffer &result,
			IntBuffer &effective_bits) {
			using namespace kernel;

			typedef cuda_memory<OutBitStream> CudaBitStreams;
			typedef cuda_memory<OutBitStream::WriteBitsInfo> CudaStreamInfos;

			const u_int buffer_size = quantized.size();
			const u_int block_num = effective_bits.size();
			const u_int block_size = buffer_size / block_num;
			const u_int mcu_num = buffer_size / 64;

			CudaByteBuffer buffer(OutBitStream::MAX_BLOCK_SIZE * mcu_num);
			CudaBitStreams stream(mcu_num);
			CudaStreamInfos info(mcu_num);

			// 各MCU用のバッファを作成
			for (u_int i = 0; i < mcu_num; ++i) {
				stream[i].setStreamBuffer(buffer.device_data() + OutBitStream::MAX_BLOCK_SIZE * i);
			}
			stream.sync_to_device();

			// バッファをクリア
			buffer.fill_zero();

			// 各MCU用ごとにエンコード
			dim3 block(CalcOptimumThreads(block_size / 6 / 64), 1, 1);
			dim3 grid(block_size / 6 / 64 / block.x, 6, block_num);
			HuffmanEncodeForMCU<<<grid, block>>>(quantized.device_data(), stream.device_data());

			// 書きこみ情報の作成
			stream.sync_to_host();
			OutBitStream::CreateWriteBitsTable(stream.host_data(), info.host_data(), mcu_num,
				mcu_num / block_num);
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
			CombineHuffmanStream<3, 0> <<<grid, block>>>(stream.device_data(), info.device_data(),
				result.device_data(), result.size() / block_num);
			CombineHuffmanStream<3, 1> <<<grid, block>>>(stream.device_data(), info.device_data(),
				result.device_data(), result.size() / block_num);
			CombineHuffmanStream<3, 2> <<<grid, block>>>(stream.device_data(), info.device_data(),
				result.device_data(), result.size() / block_num);
		}

		void HuffmanDecode(const ByteBuffer &huffman, IntBuffer &quantized, size_t width, size_t height) {
			InBitStream ibs(huffman.data(), huffman.size());
			cpu::decode_huffman(&ibs, quantized.data(), width, height);

		}

		//-------------------------------------------------------------------------------------------------//
		//
		// 符号化クラス
		//
		//-------------------------------------------------------------------------------------------------//
		class Encoder::Impl {
		private:
			typedef cuda_memory<kernel::OutBitStream> CudaBitStreams;
			typedef cuda_memory<kernel::OutBitStream::WriteBitsInfo> CudaStreamInfos;

			u_int width_;
			u_int height_;
			u_int block_width_;
			u_int block_height_;

			u_int buffer_size_;
			u_int block_size_;

			u_int quality_;

			DeviceTable encode_table_;

			DeviceByteBuffer encode_yuv_result_;
			DeviceIntBuffer encode_dct_result_;
			DeviceIntBuffer encode_qua_result_;

			DeviceByteBuffer encode_src_;

			u_int mcu_kernel_block_x_;
			u_int combine_kernel_block_x_;

			DeviceByteBuffer huffman_buffer_;
			CudaBitStreams huffman_stream_;
			CudaStreamInfos huffman_info_;

		public:
			Impl(u_int width, u_int height, u_int block_width, u_int block_height) :
				width_(width),
				height_(height),
				block_width_(block_width),
				block_height_(block_height),
				buffer_size_(width * height * 3 / 2),
				block_size_(block_width * block_height * 3 / 2),
				quality_(80),
				encode_table_(width * height),
				encode_yuv_result_(buffer_size_),
				encode_dct_result_(buffer_size_),
				encode_qua_result_(buffer_size_),
				encode_src_(width_ * height_ * 3),
				huffman_buffer_(kernel::OutBitStream::MAX_BLOCK_SIZE * buffer_size_ / 64),
				huffman_stream_(buffer_size_ / 64),
				huffman_info_(buffer_size_ / 64) {

				mcu_kernel_block_x_ = CalcOptimumThreads(block_size_ / 6 / 64);
				combine_kernel_block_x_ = CalcOptimumThreads(getMcuNum() / getBlockNum() / 3);

				// 各MCU用のバッファを作成
				for (u_int i = 0; i < getMcuNum(); ++i) {
					huffman_stream_[i].setStreamBuffer(
						huffman_buffer_.device_data() + kernel::OutBitStream::MAX_BLOCK_SIZE * i);
				}
				huffman_stream_.sync_to_device();
				huffman_buffer_.fill_zero();

				CreateConversionTable(width_, height_, block_width_, block_height_, encode_table_);
			}

			~Impl() {
#ifdef DEBUG
				DebugLog::log("Encoder::Impl::~Impl()");
#endif
			}

			void reset() {
				encode_table_.resize(width_ * height_, true);
				CreateConversionTable(width_, height_, block_width_, block_height_, encode_table_);

				buffer_size_ = width_ * height_ * 3 / 2;
				block_size_ = block_width_ * block_height_ * 3 / 2;

				encode_yuv_result_.resize(buffer_size_);
				encode_dct_result_.resize(buffer_size_);
				encode_qua_result_.resize(buffer_size_);
				encode_src_.resize(width_ * height_ * 3);

				huffman_buffer_.resize(kernel::OutBitStream::MAX_BLOCK_SIZE * buffer_size_ / 64);
				huffman_stream_.resize(buffer_size_ / 64);
				huffman_info_.resize(buffer_size_ / 64);

				// 各MCU用のバッファを作成
				for (u_int i = 0; i < getMcuNum(); ++i) {
					huffman_stream_[i].setStreamBuffer(
						huffman_buffer_.device_data() + kernel::OutBitStream::MAX_BLOCK_SIZE * i);
				}
				huffman_stream_.sync_to_device();
				huffman_buffer_.fill_zero();

				mcu_kernel_block_x_ = CalcOptimumThreads(block_size_ / 6 / 64);
				combine_kernel_block_x_ = CalcOptimumThreads(getMcuNum() / getBlockNum() / 3);
			}

			void setImageSize(u_int width, u_int height) {
				width_ = width;
				height_ = height;
			}

			void setBlockSize(u_int block_width, u_int block_height) {
				block_width_ = block_width;
				block_height_ = block_height;
			}

			void setQuality(u_int quality) {
				quality_ = quality;
			}

			u_int getBlockNum() const {
				return width_ * height_ / (block_width_ * block_height_);
			}

			u_int getBlocksPerRow() const {
				return width_ / block_width_;
			}

			u_int getMcuNum() const {
				return huffman_info_.size();
			}

			void encode(const byte* rgb, DeviceByteBuffer &huffman, IntBuffer &effective_bits) {
				assert(effective_bits.size() >= getBlockNum());

				DeviceByteBuffer encode_src(rgb, width_ * height_ * 3);
				huffman.fill_zero();

				cuda::ConvertRGBToYUV(encode_src, encode_yuv_result_, width_, height_, block_width_,
					block_height_, encode_table_);
				cuda::DiscreteCosineTransform(encode_yuv_result_, encode_dct_result_);
				cuda::ZigzagQuantize(encode_dct_result_, encode_qua_result_, block_size_, quality_);

				huffmanEncode(encode_qua_result_, huffman, effective_bits);
			}

			void encode(const DeviceByteBuffer &rgb, DeviceByteBuffer &huffman, IntBuffer &effective_bits) {
				assert(effective_bits.size() >= getBlockNum());

				CudaStopWatch watch;

				watch.start();
				huffman.fill_zero();
				watch.stop();
				std::cout << "Pre-Process, " << watch.getLastElapsedTime() * 1000.0 << std::endl;
				watch.clear();

				watch.start();
				cuda::ConvertRGBToYUV(rgb, encode_yuv_result_, width_, height_, block_width_, block_height_,
					encode_table_);
				watch.stop();
				std::cout << "Color Conversion, " << watch.getLastElapsedTime() * 1000.0 << std::endl;
				watch.clear();

				watch.start();
				cuda::DiscreteCosineTransform(encode_yuv_result_, encode_dct_result_);
				watch.stop();
				std::cout << "DCT, " << watch.getLastElapsedTime() * 1000.0 << std::endl;
				watch.clear();

				watch.start();
				cuda::ZigzagQuantize(encode_dct_result_, encode_qua_result_, block_size_, quality_);
				watch.stop();
				std::cout << "Zigzag Quantize, " << watch.getLastElapsedTime() * 1000.0 << std::endl;
				watch.clear();

				watch.start();
				huffmanEncode(encode_qua_result_, huffman, effective_bits);
				watch.stop();
				std::cout << "Huffman Encode, " << watch.getLastElapsedTime() * 1000.0 << std::endl;
				watch.clear();
			}

		private:
			void huffmanEncode(const DeviceIntBuffer &quantized, DeviceByteBuffer &result,
				IntBuffer &effective_bits) {
				using namespace kernel;

				const u_int block_num = getBlockNum();
				const u_int mcu_num = getMcuNum();

				// バッファをクリア
				huffman_buffer_.fill_zero();

				// 各MCU用ごとにエンコード
				dim3 block(mcu_kernel_block_x_, 1, 1);
				dim3 grid(block_size_ / 6 / 64 / block.x, 6, block_num);
				HuffmanEncodeForMCU<<<grid, block>>>(quantized.device_data(), huffman_stream_.device_data());

				// 書きこみ情報の作成
				huffman_stream_.sync_to_host();
				OutBitStream::CreateWriteBitsTable(huffman_stream_.host_data(), huffman_info_.host_data(),
					mcu_num, mcu_num / block_num);
				huffman_info_.sync_to_device();

				// 各画像ブロックの有効bit数を算出
				for (u_int i = 0; i < block_num; ++i) {
					u_int last_index = (i + 1) * mcu_num / block_num - 1;
					effective_bits[i] = huffman_info_[last_index].bits_of_grobal
						+ huffman_info_[last_index].bits_of_stream;
				}

				// 適切なスレッド数でMCUbitを結合
				u_int total_thread = mcu_num / block_num / 3;
				block = dim3(combine_kernel_block_x_, 1, 1);
				grid = dim3(total_thread / block.x, block_num, 1);
				CombineHuffmanStream<3, 0> <<<grid, block>>>(huffman_stream_.device_data(),
					huffman_info_.device_data(), result.device_data(), result.size() / block_num);
				CombineHuffmanStream<3, 1> <<<grid, block>>>(huffman_stream_.device_data(),
					huffman_info_.device_data(), result.device_data(), result.size() / block_num);
				CombineHuffmanStream<3, 2> <<<grid, block>>>(huffman_stream_.device_data(),
					huffman_info_.device_data(), result.device_data(), result.size() / block_num);
			}
		};

		Encoder::Encoder(u_int width, u_int height) :
			impl(new Impl(width, height, width, height)) {
		}

		Encoder::Encoder(u_int width, u_int height, u_int block_width, u_int block_height) :
			impl(new Impl(width, height, block_width, block_height)) {
		}

		Encoder::~Encoder() {
#ifdef DEBUG
			DebugLog::log("Encoder::~Encoder()");
#endif
		}

		void Encoder::reset() {
			impl->reset();
		}

		void Encoder::setImageSize(u_int width, u_int height) {
			impl->setImageSize(width, height);
		}

		void Encoder::setBlockSize(u_int block_width, u_int block_height) {
			impl->setBlockSize(block_width, block_height);
		}

		void Encoder::setQuality(u_int quality) {
			impl->setQuality(quality);
		}

		u_int Encoder::getBlockNum() const {
			return impl->getBlockNum();
		}

		u_int Encoder::getBlocksPerRow() const {
			return impl->getBlocksPerRow();
		}

		u_int Encoder::getMcuNum() const {
			return impl->getMcuNum();
		}

		void Encoder::encode(const byte* rgb, DeviceByteBuffer &huffman, IntBuffer &effective_bits) {
			impl->encode(rgb, huffman, effective_bits);
		}

		void Encoder::encode(const DeviceByteBuffer &rgb, DeviceByteBuffer &huffman,
			IntBuffer &effective_bits) {
			impl->encode(rgb, huffman, effective_bits);
		}

		class Decoder::Impl {
		private:
			u_int width_;
			u_int height_;

			u_int quality_;

			u_int buffer_size_;

			DeviceTable decode_table_;

			DeviceIntBuffer decode_dct_src_;
			DeviceByteBuffer decode_yuv_src_;
			DeviceByteBuffer decode_result_;

			CudaIntBuffer decode_qua_src_;

		public:
			Impl(u_int width, u_int height) :
				width_(width),
				height_(height),
				quality_(80),
				buffer_size_(width * height * 3 / 2),
				decode_table_(width * height),
				decode_dct_src_(buffer_size_),
				decode_yuv_src_(buffer_size_),
				decode_result_(width * height * 3),
				decode_qua_src_(buffer_size_) {

				CreateConversionTable(width, height, width, height, decode_table_);
			}

			~Impl() {
#ifdef DEBUG
				DebugLog::log("Decoder::Impl::~Impl()");
#endif
			}

			void reset() {
				CreateConversionTable(width_, height_, width_, height_, decode_table_);
			}

			void setImageSize(u_int width, u_int height) {
				width_ = width;
				height_ = height;
			}

			void setQuality(u_int quality) {
				quality_ = quality;
			}

			void decode(const byte *huffman, byte *dst) {

				CudaStopWatch watch;

				watch.start();
				InBitStream ibs(huffman, buffer_size_);
				cpu::decode_huffman(&ibs, decode_qua_src_.host_data(), width_, height_);
				watch.stop();
				std::cout << "Huffman Decode, " << watch.getLastElapsedTime() * 1000.0 << std::endl;
				watch.clear();

				watch.start();
				decode_qua_src_.sync_to_device();
				watch.stop();
				std::cout << "Memory Transfer, " << watch.getLastElapsedTime() * 1000.0 << std::endl;
				watch.clear();

				watch.start();
				cuda::InverseZigzagQuantize(decode_qua_src_, decode_dct_src_, buffer_size_, quality_);
				watch.stop();
				std::cout << "Zigzag Quantize, " << watch.getLastElapsedTime() * 1000.0 << std::endl;
				watch.clear();

				watch.start();
				cuda::InverseDiscreteCosineTransform(decode_dct_src_, decode_yuv_src_);
				watch.stop();
				std::cout << "iDCT, " << watch.getLastElapsedTime() * 1000.0 << std::endl;
				watch.clear();

				watch.start();
				cuda::ConvertYUVToRGB(decode_yuv_src_, decode_result_, width_, height_, width_, height_,
					decode_table_);
				watch.stop();
				std::cout << "Color Conversion, " << watch.getLastElapsedTime() * 1000.0 << std::endl;
				watch.clear();

				watch.start();
				decode_result_.copy_to_host(dst, decode_result_.size(), 0);
				watch.stop();
				std::cout << "Memory Transfer, " << watch.getLastElapsedTime() * 1000.0 << std::endl;
				watch.clear();
			}
		};

		Decoder::Decoder(u_int width, u_int height) :
			impl(new Impl(width, height)) {
		}

		Decoder::~Decoder() {
#ifdef DEBUG
			DebugLog::log("Decoder::~Decoder()");
#endif
		}

		void Decoder::reset() {
			impl->reset();
		}

		void Decoder::setImageSize(u_int width, u_int height) {
			impl->setImageSize(width, height);
		}

		void Decoder::setQuality(u_int quality) {
			impl->setQuality(quality);
		}

		void Decoder::decode(const byte *huffman, byte *dst) {
			impl->decode(huffman, dst);
		}
	} // namespace cuda
} // namespace jpeg
