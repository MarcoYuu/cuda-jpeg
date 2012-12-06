#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <jpeg/cpu/cpu_jpeg.h>
#include <jpeg/cuda/cuda_jpeg.cuh>

#include <utils/debug_log.h>
#include <utils/type_definitions.h>
#include <utils/utility.hpp>
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
				const u_int pix_index = threadIdx.x + threadIdx.y * blockDim.x
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

				u_int x = threadIdx.x, y = threadIdx.y;
				u_int local_index = x + y * 8;
				u_int start_index = 64 * blockIdx.x;

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

				u_int x = threadIdx.x, y = threadIdx.y;
				u_int local_index = x + y * 8;
				u_int start_index = 64 * blockIdx.x;

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

				u_int local_index = threadIdx.x + threadIdx.y * 8;
				u_int start_index = 64 * threadIdx.z + 128 * blockIdx.x + 128 * gridDim.x * blockIdx.y
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

				u_int local_index = threadIdx.x + threadIdx.y * 8;
				u_int start_index = 64 * threadIdx.z + 128 * blockIdx.x + 128 * gridDim.x * blockIdx.y
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

				u_int local_index = threadIdx.x + threadIdx.y * 8;
				u_int start_index = 64 * threadIdx.z + 128 * blockIdx.x + 128 * gridDim.x * blockIdx.y
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

				u_int local_index = threadIdx.x + threadIdx.y * 8;
				u_int start_index = 64 * threadIdx.z + 128 * blockIdx.x + 128 * gridDim.x * blockIdx.y
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
			u_int block_size, u_int quarity) {
			// 最低品質
			if (quarity == 0) {
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
			u_int block_size, u_int quarity) {
			// 最低品質
			if (quarity == 0) {
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
	} // namespace cuda
} // namespace jpeg
