#include <cstdio>
#include <cstdlib>

#include <iostream>

#include <jpeg/cuda/cuda_jpeg.cuh>
#include <utils/cuda/bit_operation.cuh>

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

//				if (src_index == 1) {
//					printf("%d, %d, %d, %d, %d, %d\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
//					printf("%d, %d, %d,\n%d, %d, %d,\n%d, %d, %d,\n%d, %d, %d\n\n",
//						blockIdx.x, blockIdx.y, blockIdx.z,
//						threadIdx.x, threadIdx.y, threadIdx.z,
//						src_block_start_index, src_mcu_start_index, src_index,
//						dst_block_start_y_index, dst_mcu_y_start_index, dst_y_index);
//					printf("%d, %d, %d, %d, %d, %d\n", dst_block_start_y_index, dst_mcu_y_start_index, dst_y_index,
//						src_block_start_index, src_mcu_start_index, src_index);
//				}
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
				const int pix_index = threadIdx.x + threadIdx.y * blockDim.x
					+ (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * blockDim.x
						* blockDim.y;

				const TableElementSrcToDst elem = table[pix_index];
				const u_int src_index = pix_index * 3;

				// R,G,B [0, 255] -> R,G,B [16, 235]
				const float b = rgb[src_index + 0] * 0.8588f + 16.0f;
				const float g = rgb[src_index + 1] * 0.8588f + 16.0f;
				const float r = rgb[src_index + 2] * 0.8588f + 16.0f;

				// R,G,B [16, 235] -> Y [16, 235] U,V [16, 240]
				yuv_result[elem.y] = 0.11448f * b + 0.58661f * g + 0.29891f * r;
				yuv_result[elem.u] = 0.50000f * b - 0.33126f * g - 0.16874f * r + 128.0f;
				yuv_result[elem.v] = -0.08131f * b - 0.41869f * g + 0.50000f * r + 128.0f;
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
				const int pix_index = threadIdx.x + threadIdx.y * blockDim.x
					+ (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * blockDim.x
						* blockDim.y;

				const TableElementSrcToDst elem = table[pix_index];
				const u_int dst_index = pix_index * 3;

				// Y [16, 235] U,V [16, 240] -> Y [16, 235] U,V [-112, 112]
				const float y = yuv[elem.y];
				const float u = yuv[elem.u] - 128.0f;
				const float v = yuv[elem.v] - 128.0f;

				// Y [16, 235] U,V [-112, 112] -> R,G,B [0, 255]
				rgb_result[dst_index + 0] = (y + 1.77200f * u - 16.0f) * 1.164f;
				rgb_result[dst_index + 1] = (y - 0.34414f * u - 0.71414f * v - 16.0f) * 1.164f;
				rgb_result[dst_index + 2] = (y + 1.40200f * v - 16.0f) * 1.164f;
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
			 * 	- grid(yuv.size() / 64, 1, 1)
			 * 	- block(8, 8, 1)
			 * - グリッド/ブロック数に制限はない
			 * - dst_coefficient.size == yuv.sizeであること
			 *
			 * @param yuv_src 64byte=8x8blockごとに連続したメモリに保存されたデータ
			 * @param dct_coefficient DCT係数
			 */
			void __global__ DiscreteCosineTransform(const byte *yuv_src, int *dct_coefficient) {
				using DCTConstants::cos;
				using DCTConstants::cos_t;

				int x = threadIdx.x, y = threadIdx.y;
				int local_index = x + y * 8;
				int start_index = 64 * blockIdx.x;

				__shared__ float vertical_result[64];

				vertical_result[local_index] = cos[y * 8 + 0] * yuv_src[start_index + x + 0 * 8]
					+ cos[y * 8 + 1] * yuv_src[start_index + x + 1 * 8]
					+ cos[y * 8 + 2] * yuv_src[start_index + x + 2 * 8]
					+ cos[y * 8 + 3] * yuv_src[start_index + x + 3 * 8]
					+ cos[y * 8 + 4] * yuv_src[start_index + x + 4 * 8]
					+ cos[y * 8 + 5] * yuv_src[start_index + x + 5 * 8]
					+ cos[y * 8 + 6] * yuv_src[start_index + x + 6 * 8]
					+ cos[y * 8 + 7] * yuv_src[start_index + x + 7 * 8];

				__syncthreads();

				dct_coefficient[start_index + local_index] = vertical_result[y * 8 + 0] * cos_t[x + 0 * 8]
					+ vertical_result[y * 8 + 1] * cos_t[x + 1 * 8]
					+ vertical_result[y * 8 + 2] * cos_t[x + 2 * 8]
					+ vertical_result[y * 8 + 3] * cos_t[x + 3 * 8]
					+ vertical_result[y * 8 + 4] * cos_t[x + 4 * 8]
					+ vertical_result[y * 8 + 5] * cos_t[x + 5 * 8]
					+ vertical_result[y * 8 + 6] * cos_t[x + 6 * 8]
					+ vertical_result[y * 8 + 7] * cos_t[x + 7 * 8];
			}

			/**
			 * @brief iDCTカーネル
			 *
			 * - カーネル起動は各YUVごと = width*height*3/2スレッド必要
			 * 	- grid(yuv.size() / 64, 1, 1)
			 * 	- block(8, 8, 1)
			 * - グリッド/ブロック数に制限はない
			 * - dst_coefficient.size == yuv.sizeであること
			 *
			 * @param dct_coefficient DCT係数
			 * @param yuv_result 64byte=8x8blockごとに連続したメモリに保存されたデータ
			 */
			void __global__ InverseDiscreteCosineTransform(const int *dct_coefficient, byte *yuv_result) {
				using DCTConstants::cos;
				using DCTConstants::cos_t;

				int x = threadIdx.x, y = threadIdx.y;
				int local_index = x + y * 8;
				int start_index = 64 * blockIdx.x;

				__shared__ float vertical_result[64];

				vertical_result[local_index] = cos_t[y * 8 + 0] * dct_coefficient[start_index + x + 0 * 8]
					+ cos_t[y * 8 + 1] * dct_coefficient[start_index + x + 1 * 8]
					+ cos_t[y * 8 + 2] * dct_coefficient[start_index + x + 2 * 8]
					+ cos_t[y * 8 + 3] * dct_coefficient[start_index + x + 3 * 8]
					+ cos_t[y * 8 + 4] * dct_coefficient[start_index + x + 4 * 8]
					+ cos_t[y * 8 + 5] * dct_coefficient[start_index + x + 5 * 8]
					+ cos_t[y * 8 + 6] * dct_coefficient[start_index + x + 6 * 8]
					+ cos_t[y * 8 + 7] * dct_coefficient[start_index + x + 7 * 8];

				__syncthreads();

				float value = vertical_result[y * 8 + 0] * cos[x + 0 * 8]
					+ vertical_result[y * 8 + 1] * cos[x + 1 * 8]
					+ vertical_result[y * 8 + 2] * cos[x + 2 * 8]
					+ vertical_result[y * 8 + 3] * cos[x + 3 * 8]
					+ vertical_result[y * 8 + 4] * cos[x + 4 * 8]
					+ vertical_result[y * 8 + 5] * cos[x + 5 * 8]
					+ vertical_result[y * 8 + 6] * cos[x + 6 * 8]
					+ vertical_result[y * 8 + 7] * cos[x + 7 * 8];

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
			 * @param quariry 量子化品質[0,100]
			 */
			void __global__ ZigzagQuantizeLow(const int *dct_coefficient, int *quantized, float quariry) {
				using Zigzag::sequence;

				int local_index = threadIdx.x + threadIdx.y * 8;
				int start_index = 64 * threadIdx.z + 128 * blockIdx.x + 128 * gridDim.x * blockIdx.y
					+ gridDim.x * 128 * 3 * blockIdx.z;
				quantized[start_index + sequence[local_index]] = dct_coefficient[start_index + local_index]
					/ ((255.0f - ((255.0f - quantize_table[blockIdx.y][local_index]) * (1.0f + quariry))));
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
			 * @param quariry 量子化品質[0,100]
			 */
			void __global__ InverseZigzagQuantizeLow(const int *quantized, int *dct_coefficient,
				float quariry) {
				using Zigzag::sequence;

				int local_index = threadIdx.x + threadIdx.y * 8;
				int start_index = 64 * threadIdx.z + 128 * blockIdx.x + 128 * gridDim.x * blockIdx.y
					+ gridDim.x * 128 * 3 * blockIdx.z;
				dct_coefficient[start_index + local_index] = quantized[start_index + sequence[local_index]]
					* ((255.0f - ((255.0f - quantize_table[blockIdx.y][local_index]) * (1.0f + quariry))));
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
			 * @param quarity 量子化品質[0,100]
			 */
			void __global__ ZigzagQuantizeHigh(const int *dct_coefficient, int *quantized, float quarity) {
				using Zigzag::sequence;

				int local_index = threadIdx.x + threadIdx.y * 8;
				int start_index = 64 * threadIdx.z + 128 * blockIdx.x + 128 * gridDim.x * blockIdx.y
					+ gridDim.x * 128 * 3 * blockIdx.z;
				quantized[start_index + sequence[local_index]] = dct_coefficient[start_index + local_index]
					/ (quantize_table[blockIdx.y][local_index] * (1.0f - quarity));
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
			 * @param quarity 量子化品質[0,100]
			 */
			void __global__ InverseZigzagQuantizeHigh(const int *quantized, int *dct_coefficient,
				float quarity) {
				using Zigzag::sequence;

				int local_index = threadIdx.x + threadIdx.y * 8;
				int start_index = 64 * threadIdx.z + 128 * blockIdx.x + 128 * gridDim.x * blockIdx.y
					+ gridDim.x * 128 * 3 * blockIdx.z;
				dct_coefficient[start_index + local_index] = quantized[start_index + sequence[local_index]]
					* (quantize_table[blockIdx.y][local_index] * (1.0f - quarity));
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

				int local_index = threadIdx.x + threadIdx.y * 8;
				int start_index = 64 * threadIdx.z + 192 * blockIdx.x;
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

				int local_index = threadIdx.x + threadIdx.y * 8;
				int start_index = 64 * threadIdx.z + 192 * blockIdx.x;
				dct_coefficient[start_index + local_index] = quantized[start_index + sequence[local_index]];
			}
			//-------------------------------------------------------------------------------------------------//
			//
			// ハフマン符号化
			//
			//-------------------------------------------------------------------------------------------------//
			//TODO 使って試してみる
			namespace huffman {
				using namespace cuda::encode_table::HuffmanEncode;

				/**
				 * 余ったビットに1を詰めるためのマスク
				 */
				static __device__ __constant__ const unsigned char kBitFullMaskT[] = {
					0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0x7f, 0xff };
				/** 余ったビットに1を詰めるためのマスク */
				static __device__ __constant__ const unsigned char kBitFullMaskLowT[] = {
					0xff, 0x7f, 0x3f, 0x1f, 0x0f, 0x07, 0x03, 0x01 };

				/**
				 * @brief CUDAによる圧縮のMCUごとのステート
				 *
				 * @author yuumomma
				 * @version 1.0
				 */
				class OutBitStream {

					int byte_pos_;
					int bit_pos_; //! 書き込みビット位置（上位ビットが7、下位ビットが0）
					int writable_; //! 1:書き込み可, 0:書き込み不可
					int num_bits_; //! 全体バッファに書き込むサイズ

					byte *dst_buf_;

				public:
					static const int MAX_BLOCK_SIZE = 128;

					/**
					 * @brief コンストラクタ
					 */
					inline __host__ OutBitStream() :
						byte_pos_(0),
						bit_pos_(7),
						writable_(1),
						num_bits_(0),
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
						num_bits_(0),
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
					inline __device__ void setBits(int value, int num_bits) {
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
					inline __device__ void setFewBits(byte value, int num_bits) {
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
					inline __device__ void setBits2Byte(byte value, int num_bits) {
						// 上位ビットをクリア
						value &= kBitFullMaskT[num_bits - 1];
						// 次のバイトに入れるビット数
						int nextBits = num_bits - (bit_pos_ + 1);

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
					inline __device__ void set8Bits(byte value, int num_bits) {
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
						int bits_of_stream; /// このバッファのbit数
						int bits_of_grobal; /// このバッファまでのbit数
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
						WriteBitsInfo* num_bits, int stream_num, int stream_per_block) {

						//BitPosは7が上位で0が下位なので注意,更に位置なので注意。7なら要素は0,0なら要素は7
						num_bits[0].bits_of_stream = stream[0].byte_pos_ * 8 + (7 - stream[0].bit_pos_);
						num_bits[0].bits_of_grobal = 0;

						int blocks = stream_num / stream_per_block;
						for (int i = 0; i < blocks; ++i) {
							//BitPosは7が上位で0が下位なので注意,更に位置なので注意。7なら要素は0,0なら要素は7
							int first_stream_of_block = i * stream_per_block;
							num_bits[first_stream_of_block].bits_of_stream = stream[0].byte_pos_ * 8
								+ (7 - stream[0].bit_pos_);
							num_bits[first_stream_of_block].bits_of_grobal = 0;

							for (int j = first_stream_of_block + 1;
								j < first_stream_of_block + stream_per_block; ++j) {
								num_bits[j].bits_of_stream = stream[j].byte_pos_ * 8
									+ (7 - stream[0].bit_pos_);
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
						const int bytes = info->bits_of_stream / 8;
						const int rest_bits = info->bits_of_stream - 8 * bytes;

						const int offset_bytes = info->bits_of_grobal / 8;
						const int offset_bits = info->bits_of_grobal - 8 * offset_bytes;

						const byte *data = stream->getData();
						for (int i = 0; i < bytes; ++i) {
							dst[i] |= ((data[i] >> offset_bits) & kBitFullMaskLowT[offset_bits]);
							dst[i + 1] |= (data[i] << (8 - offset_bits));
						}
						dst[bytes] |= ((data[bytes] >> offset_bits)
							& ~(kBitFullMaskLowT[offset_bits] | (0xff >> (rest_bits + offset_bits))));
						dst[bytes + 1] |= (data[bytes] << (8 - offset_bits));
					}
				};

				static __device__ __constant__ const int* DCSizeTable[] = {
					DC::luminance::code_size, DC::luminance::code_size, DC::component::code_size };

				static __device__ __constant__ const int* DCCodeTable[] = {
					DC::luminance::code, DC::luminance::code, DC::component::code };

				static __device__ __constant__ const int* ACSizeTable[] = {
					AC::luminance::code_size, AC::luminance::code_size, AC::component::code_size };

				static __device__ __constant__ const int* ACCodeTable[] = {
					AC::luminance::code, AC::luminance::code, AC::component::code };

				/**
				 * @brief MCU(8x8)毎ハフマン符号化カーネル
				 *
				 * - 基本的に[huffman_code][value][huffman_code][value]…と続く
				 * - カーネル呼び出しは8x8ブロックごと = (width*height*3/2)/64 thread.
				 * - 最低は16x16の画像で、8X8ブロックは6個 = 最低 16x16x3/2/64 : 6 thread.
				 * - blockIDx.yでわけるとするとTHREADSは最大 buffer_size/64/3/block_num.
				 *
				 *	- buffer_size = width * height * 3 / 2;
				 * 	- block_size = block_width * block_height * 3 / 2;
				 * 	- block_num = buffer_size / block_size;
				 * 	- THREADS = block_size / 64 / 3;
				 *
				 * 	- grid(block_size / 64 / 3 / THREADS, 3, block_num)
				 * 	- block(THREADS, 1, 1)
				 *
				 * @param quantized 量子化データ
				 * @param dst 書き込み先
				 *
				 */__global__ void HuffmanEncodeForMCU(const int *quantized, OutBitStream *dst) {
					using namespace encode_table::HuffmanEncode;

					// マクロブロック番号
					const int mcu_id = threadIdx.x + blockDim.x * blockIdx.x
						+ gridDim.x * blockDim.x * blockIdx.y + gridDim.x * 3 * blockIdx.z;

					// 量子化結果におけるマクロブロックの開始インデックス
					const int mcu_start_index = 64 * mcu_id;

					// マクロブロックごとのバッファ
					OutBitStream *dst_mcu = &dst[mcu_id];

					// 各画像ブロックの左上で有るかどうか(左上なら0,そうでないなら1)
					const int is_block_left_top = (int) (!(threadIdx.x == 0 && blockIdx.x == 0
						&& blockIdx.y == 0));

					// ----------------------------- DC成分 ------------------------------------
					// DC成分は前のMCUの成分との差分をハフマン符号化するため
					// 画像の最も左上のMCUとの差分は0と取る
					int diff = quantized[mcu_start_index]
						- is_block_left_top * quantized[mcu_start_index - 64 * is_block_left_top];
					//int diff = 0;
					//if (mcu_start_index == 0) {
					//	diff = quantized[mcu_start_index];
					//} else {
					//	diff = quantized[mcu_start_index] - quantized[mcu_start_index - 64];
					//}

					// 差分の絶対値から最低限その値を表すのに必要なbitを計算
					// おそらく有効bitは11(=2^11=2048以下)のはず
					byte4 abs_diff = abs(diff);
					int effective_bits = EffectiveBits(abs_diff);
					dst_mcu->setBits(DCCodeTable[blockIdx.y][effective_bits],
						DCSizeTable[blockIdx.y][effective_bits]);

					// 0以外ならハフマンbitに続けて実値を書き込む
					if (effective_bits != 0) {
						//if (diff < 0)
						//	--diff;
						diff -= (int) (diff < 0);
						dst_mcu->setBits(diff, effective_bits);
					}

					// ----------------------------- AC成分 ------------------------------------
					// 残り63のDCT係数に対して処理を行う
					int runlength = 0;
#pragma unroll
					// 末尾以外
					for (int i = 1; i < 63; i++) {
						int value = quantized[mcu_start_index + i];
						byte4 abs_value = abs(value);

						if (abs_value != 0) {
							// 0の個数が16ごとにZRL=code_id:151を代わりに割り当てる
							while (runlength > 15) {
								dst_mcu->setBits(ACCodeTable[blockIdx.y][AC::luminance::ZRL],
									ACSizeTable[blockIdx.y][AC::luminance::ZRL]);
								runlength -= 16;
							}

							// 有効bit数と15以下の0の個数を合わせて符号化コードを書き込む
							// おそらくAC成分の有効bitは10(=2^10=1024)以下のはず
							// したがってcode_idは[1,150][152,161]の範囲
							effective_bits = EffectiveBits(abs_value);
							int code_id = runlength * 10 + effective_bits + (runlength == 15);
							dst_mcu->setBits(ACCodeTable[blockIdx.y][code_id],
								ACSizeTable[blockIdx.y][code_id]);
							runlength = 0;

							// ハフマンbitに続けて実値を書き込む
							//if (value < 0)
							//	--value;
							value -= (int) (value < 0);
							dst_mcu->setBits(value, effective_bits);

						} else {
							++runlength;
						}
					}
					// 末尾はEOB=code_id:0
					dst_mcu->setBits(AC::luminance::code[AC::luminance::EOB],
						AC::luminance::code_size[AC::luminance::EOB]);
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
				template<int Scale, int Stride>
				__global__ void CombineHuffmanStream(const OutBitStream *stream,
					const OutBitStream::WriteBitsInfo *info, byte *dst, u_int block_size) {
					int mcu_id = (blockIdx.x * blockDim.x + threadIdx.x) * Scale + Stride;
					int block_id = blockIdx.y;

					OutBitStream::WriteStreamToLineBuffer(stream + mcu_id, info + mcu_id,
						dst + block_size * block_id);
				}
			} // namespace huffman
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

			kernel::CreateConversionTable<<<grid,block>>>(width, height, block_width,
				block_height, table.device_data());
		}

		void ConvertRGBToYUV(const DeviceByteBuffer &rgb, DeviceByteBuffer &yuv_result, size_t width,
			size_t height, size_t block_width, size_t block_height, const DeviceTable &table) {
			assert(rgb.size()/2 <= yuv_result.size());

			const dim3 grid(block_width / 16, block_height / 16, width / block_width * height / block_height);
			const dim3 block(16, 16, 1);

			kernel::ConvertRGBToYUV<<<grid,block>>>(rgb.device_data(), yuv_result.device_data(), table.device_data());
		}

		void ConvertYUVToRGB(const DeviceByteBuffer &yuv, DeviceByteBuffer &rgb_result, size_t width,
			size_t height, size_t block_width, size_t block_height, const DeviceTable &table) {
			assert(yuv.size() <= rgb_result.size()/2);

			const dim3 grid(block_width / 16, block_height / 16, width / block_width * height / block_height);
			const dim3 block(16, 16, 1);

			kernel::ConvertYUVToRGB<<<grid,block>>>(yuv.device_data(), rgb_result.device_data(), table.device_data());
		}

		void DiscreteCosineTransform(const DeviceByteBuffer &yuv, DeviceIntBuffer &dct_coefficient) {
			// grid (8x8ブロックの個数, 分割数, 1)
			const dim3 grid(yuv.size() / 64, 1, 1);
			// grid (1, 8x8ブロック)
			const dim3 block(8, 8, 1);

			kernel::DiscreteCosineTransform<<<grid,block>>>(
				yuv.device_data(), dct_coefficient.device_data());
		}

		void InverseDiscreteCosineTransform(const DeviceIntBuffer &dct_coefficient,
			DeviceByteBuffer &yuv_result) {
			// grid (8x8ブロックの個数, 分割数, 1)
			const dim3 grid(dct_coefficient.size() / 64, 1, 1);
			// grid (1, 8x8ブロック)
			const dim3 block(8, 8, 1);

			kernel::InverseDiscreteCosineTransform<<<grid,block>>>(
				dct_coefficient.device_data(), yuv_result.device_data());
		}

		void CalculateDCTMatrix(float *dct_mat) {
			const float PI = 3.141592653589793f;
			for (int k = 0; k < 8; ++k) {
				for (int n = 0; n < 8; ++n) {
					float c = (k == 0) ? 1.0f / 1.41421356f : 1.0f;
					int index = n + k * 8;
					dct_mat[index] = c * 0.5f * cos((2.0f * n + 1) * k * PI / 16.0f);
				}
			}
		}

		void CalculateiDCTMatrix(float *idct_mat) {
			const float PI = 3.141592653589793f;
			for (int n = 0; n < 8; ++n) {
				for (int k = 0; k < 8; ++k) {
					float c = (n == 0) ? 1.0f / 1.41421356f : 1.0f;
					int index = n + k * 8;
					idct_mat[index] = c * 0.5f * cos((2 * k + 1) * n * PI / 16.0f);
				}
			}
		}

		void ZigzagQuantize(const DeviceIntBuffer &dct_coefficient, DeviceIntBuffer &quantized,
			int block_size, int quarity) {
			// 最低品質
			if (quarity < 0) {
				const dim3 grid(quantized.size() / 64 / 3, 3, 1);
				const dim3 block(8, 8, 1);
				kernel::ZigzagQuantizeLow<<<grid,block>>>(
					dct_coefficient.device_data(), quantized.device_data(), -1.0f);
			}
			// 低品質
			else if (quarity < 50) {
				const dim3 grid(block_size / 128 / 3, 3, dct_coefficient.size() / block_size);
				const dim3 block(8, 8, 2);
				kernel::ZigzagQuantizeLow<<<grid,block>>>(
					dct_coefficient.device_data(), quantized.device_data(), (quarity - 50.0f) / 50.0f);
			}
			// 高品質
			else if (quarity < 100) {
				const dim3 grid(block_size / 128 / 3, 3, dct_coefficient.size() / block_size);
				const dim3 block(8, 8, 2);
				kernel::ZigzagQuantizeHigh<<<grid,block>>>(
					dct_coefficient.device_data(), quantized.device_data(), (quarity - 50.0f) / 50.0f);
			}
			// 最高品質
			else {
				const dim3 grid(quantized.size() / 192, 3, 1);
				const dim3 block(8, 8, 3);
				kernel::ZigzagQuantizeMax<<<grid,block>>>(
					dct_coefficient.device_data(), quantized.device_data());
			}
		}

		void InverseZigzagQuantize(const DeviceIntBuffer &quantized, DeviceIntBuffer &dct_coefficient,
			int block_size, int quarity) {
			// 最低品質
			if (quarity < 0) {
				const dim3 grid(quantized.size() / 64 / 3, 3, 1);
				const dim3 block(8, 8, 1);
				kernel::InverseZigzagQuantizeLow<<<grid,block>>>(
					quantized.device_data(), dct_coefficient.device_data(), -1.0f);
			}
			// 低品質
			else if (quarity < 50) {
				const dim3 grid(block_size / 128 / 3, 3, dct_coefficient.size() / block_size);
				const dim3 block(8, 8, 2);
				kernel::InverseZigzagQuantizeLow<<<grid,block>>>(
					quantized.device_data(), dct_coefficient.device_data(), (quarity - 50.0f) / 50.0f);
			}
			// 高品質
			else if (quarity < 100) {
				const dim3 grid(block_size / 128 / 3, 3, dct_coefficient.size() / block_size);
				const dim3 block(8, 8, 2);
				kernel::InverseZigzagQuantizeHigh<<<grid,block>>>(
					quantized.device_data(), dct_coefficient.device_data(), (quarity - 50.0f) / 50.0f);
			}
			// 最高品質
			else {
				const dim3 grid(quantized.size() / 192, 1, 1);
				const dim3 block(8, 8, 3);
				kernel::InverseZigzagQuantizeMax<<<grid,block>>>(
					quantized.device_data(), dct_coefficient.device_data());
			}
		}

		void HuffmanEncode(const DeviceIntBuffer &quantized, DeviceByteBuffer &result, IntBuffer &effective_bits) {
			using namespace kernel::huffman;

			const int buffer_size = quantized.size();
			const int block_num = effective_bits.size();
			const int block_size = buffer_size / block_num;
			const int mcu_num = buffer_size / 64;

			DeviceByteBuffer buffer(OutBitStream::MAX_BLOCK_SIZE * mcu_num);
			cuda_memory<OutBitStream> stream(mcu_num);
			cuda_memory<OutBitStream::WriteBitsInfo> info(mcu_num);
			for (int i = 0; i < mcu_num; ++i) {
				stream[i].setStreamBuffer(buffer.device_data() + OutBitStream::MAX_BLOCK_SIZE * i);
			}
			stream.sync_to_device();

			dim3 block(buffer_size / 64 / 3 / block_num, 1, 1);
			dim3 grid(block_size / 64 / 3 / block.x, 3, block_num);
			HuffmanEncodeForMCU<<<grid,block>>>(quantized.device_data(), stream.device_data());

			stream.sync_to_host();
			OutBitStream::CreateWriteBitsTable(stream.host_data(), info.host_data(), mcu_num, mcu_num / block_num);
			info.sync_to_device();

			for (int i = 0; i < block_num; ++i) {
				effective_bits[i] = info[i * block_size - 1].bits_of_grobal + info[i * block_size - 1].bits_of_stream;
			}

			int thread_devide = 1;
			int threadx = mcu_num / block_num / 3;
			int i = 1;
			while (i < threadx) {
				int thread_per_block = threadx / i;

				if (thread_per_block < 32 || thread_per_block) {
					break;
				}
				if (thread_per_block == 32) {
					thread_devide = i;
				}
				else if (thread_per_block == 64) {
					thread_devide = i;
				}
				else if (thread_per_block == 128) {
					thread_devide = i;
				}
				else if (thread_per_block == 192) {
					thread_devide = i;
				}
				else if (thread_per_block == 256) {
					thread_devide = i;
				}
			}
			grid = dim3(thread_devide, block_num, 1);
			block = dim3(threadx / thread_devide, 1, 1);

			CombineHuffmanStream<3, 0> <<<grid,block>>>(
				stream.device_data(), info.device_data(), result.device_data(), block_size);
			CombineHuffmanStream<3, 1> <<<grid,block>>>(
				stream.device_data(), info.device_data(), result.device_data(), block_size);
			CombineHuffmanStream<3, 2> <<<grid,block>>>(
				stream.device_data(), info.device_data(), result.device_data(), block_size);
		}
	} // namespace cuda
} // namespace jpeg
