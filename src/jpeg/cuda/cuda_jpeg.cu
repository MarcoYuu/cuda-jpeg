#include "stdio.h"
#include "stdlib.h"
#include "cuda_jpeg.cuh"
#include <iostream>

#include "../../utils/cuda/bit_operation.cuh"
#include "../ohmura/encoder_tables_device.cuh"

namespace jpeg {
	namespace cuda {

		using namespace util;
		using namespace util::cuda;
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
			// 定数
			//
			//-------------------------------------------------------------------------------------------------//
			/**
			 * DCT用定数
			 */
			namespace DCTConstants {
				/**
				 * 8x8DCTの変換係数行列
				 *
				 */__device__ __constant__ static const float CosT[] = {
					0.35355338, 0.35355338, 0.35355338, 0.35355338, 0.35355338, 0.35355338, 0.35355338,
					0.35355338, 0.49039263, 0.41573480, 0.27778509, 0.09754512, -0.09754516, -0.27778518,
					-0.41573483, -0.49039266, 0.46193978, 0.19134171, -0.19134176, -0.46193978, -0.46193978,
					-0.19134156, 0.19134180, 0.46193978, 0.41573480, -0.09754516, -0.49039266, -0.27778500,
					0.27778521, 0.49039263, 0.09754504, -0.41573489, 0.35355338, -0.35355338, -0.35355332,
					0.35355350, 0.35355338, -0.35355362, -0.35355327, 0.35355341, 0.27778509, -0.49039266,
					0.09754521, 0.41573468, -0.41573489, -0.09754511, 0.49039266, -0.27778542, 0.19134171,
					-0.46193978, 0.46193978, -0.19134195, -0.19134149, 0.46193966, -0.46193987, 0.19134195,
					0.09754512, -0.27778500, 0.41573468, -0.49039260, 0.49039271, -0.41573480, 0.27778557,
					-0.09754577 };
				/**
				 * 8x8DCTの変換係数転地行列
				 *
				 */__device__ __constant__ static const float TransposedCosT[] = {
					0.35355338, 0.49039263, 0.46193978, 0.41573480, 0.35355338, 0.27778509, 0.19134171,
					0.09754512, 0.35355338, 0.41573480, 0.19134171, -0.09754516, -0.35355338, -0.49039266,
					-0.46193978, -0.27778500, 0.35355338, 0.27778509, -0.19134176, -0.49039266, -0.35355332,
					0.09754521, 0.46193978, 0.41573468, 0.35355338, 0.09754512, -0.46193978, -0.27778500,
					0.35355350, 0.41573468, -0.19134195, -0.49039260, 0.35355338, -0.09754516, -0.46193978,
					0.27778521, 0.35355338, -0.41573489, -0.19134149, 0.49039271, 0.35355338, -0.27778518,
					-0.19134156, 0.49039263, -0.35355362, -0.09754511, 0.46193966, -0.41573480, 0.35355338,
					-0.41573483, 0.19134180, 0.09754504, -0.35355327, 0.49039266, -0.46193987, 0.27778557,
					0.35355338, -0.49039266, 0.46193978, -0.41573489, 0.35355341, -0.27778542, 0.19134195,
					-0.09754577 };
			} // namespace DCTConstants

			/**
			 * ジグザグシーケンス用定数
			 */
			namespace Zigzag {
				/**
				 * ジグザグシーケンス用
				 *
				 */__device__ __constant__ static const int sequence[] = {
					0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9,
					11, 18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55,
					60, 21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63 };
			} // namespace Zigzag

			/**
			 * 量子化テーブル用定数
			 */
			namespace quantize {
				/**
				 * 輝度用
				 *
				 */__device__ __constant__ static const int luminance[] = {
					16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57,
					69, 56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64,
					81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99 };
				/**
				 * 色差用
				 *
				 */__device__ __constant__ static const int component[] = {
					17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99,
					99, 99, 47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
					99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99 };

				/**
				 * 量子化用のテーブル
				 *
				 */__device__ __constant__ static const int* YUV[] = { luminance, luminance, component };
			} // namespace quantize

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
			 * @param width もと画像の幅
			 * @param height 元画像の高さ
			 * @param block_width ブロックの幅
			 * @param block_height ブロックの高さ
			 * @param table テーブル出力
			 *
			 */__global__ void CreateConversionTable(u_int width, u_int height, u_int block_width,
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

				const u_int mcu_y = blockIdx.x;
				const u_int mcu_x = blockIdx.y;
				const u_int mcu_id = mcu_x + mcu_y * mcu_block_x_num;
				const u_int src_mcu_start_index = src_block_start_index + mcu_y * width * 16 + mcu_x * 16;
				const u_int dst_mcu_start_y_index = dst_block_start_y_index + mcu_id * 256;
				const u_int dst_mcu_u_start_index = dst_block_start_u_index + mcu_id * 64;
				const u_int dst_mcu_v_start_index = dst_block_start_v_index + mcu_id * 64;

				const u_int pix_y = threadIdx.x;
				const u_int pix_x = threadIdx.y;

				const u_int mcu_id_x = pix_x / 8; // 0,1
				const u_int mcu_id_y = pix_y / 8; // 0,1
				const u_int block_8x8_id = mcu_id_x + 2 * mcu_id_y; // 0-3
				const u_int dst_mcu_y_8x8_index = pix_x % 8 + (pix_y % 8) * 8; // 0-63
				const u_int x = pix_x / 2, y = pix_y / 2; // 0-63

				// RGB画像のピクセルインデックス
				const u_int src_index = src_mcu_start_index + pix_x + pix_y * width;
				// YUVの書き込みインデックス
				const u_int dst_y_index = dst_mcu_start_y_index + block_8x8_id * 64 + dst_mcu_y_8x8_index;
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
			 *
			 */__global__ void ConvertRGBToYUV(const byte* rgb, byte* yuv_result,
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
			 *
			 */__global__ void ConvertYUVToRGB(const byte* yuv, byte* rgb_result,
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
			 *
			 */__global__ void DiscreteCosineTransform(const byte *yuv_src, int *dct_coefficient) {
				using DCTConstants::CosT;
				using DCTConstants::TransposedCosT;

				int x = threadIdx.x, y = threadIdx.y;
				int local_index = x + y * 8;
				int start_index = 64 * blockIdx.x;

				__shared__ float vertical_result[64];

				vertical_result[local_index] = CosT[y * 8 + 0] * yuv_src[start_index + x + 0 * 8]
					+ CosT[y * 8 + 1] * yuv_src[start_index + x + 1 * 8]
					+ CosT[y * 8 + 2] * yuv_src[start_index + x + 2 * 8]
					+ CosT[y * 8 + 3] * yuv_src[start_index + x + 3 * 8]
					+ CosT[y * 8 + 4] * yuv_src[start_index + x + 4 * 8]
					+ CosT[y * 8 + 5] * yuv_src[start_index + x + 5 * 8]
					+ CosT[y * 8 + 6] * yuv_src[start_index + x + 6 * 8]
					+ CosT[y * 8 + 7] * yuv_src[start_index + x + 7 * 8];

				__syncthreads();

				dct_coefficient[start_index + local_index] = vertical_result[y * 8 + 0]
					* TransposedCosT[x + 0 * 8] + vertical_result[y * 8 + 1] * TransposedCosT[x + 1 * 8]
					+ vertical_result[y * 8 + 2] * TransposedCosT[x + 2 * 8]
					+ vertical_result[y * 8 + 3] * TransposedCosT[x + 3 * 8]
					+ vertical_result[y * 8 + 4] * TransposedCosT[x + 4 * 8]
					+ vertical_result[y * 8 + 5] * TransposedCosT[x + 5 * 8]
					+ vertical_result[y * 8 + 6] * TransposedCosT[x + 6 * 8]
					+ vertical_result[y * 8 + 7] * TransposedCosT[x + 7 * 8];
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
			 *
			 */__global__ void InverseDiscreteCosineTransform(const int *dct_coefficient, byte *yuv_result) {
				using DCTConstants::CosT;
				using DCTConstants::TransposedCosT;

				int x = threadIdx.x, y = threadIdx.y;
				int local_index = x + y * 8;
				int start_index = 64 * blockIdx.x;

				__shared__ float vertical_result[64];

				vertical_result[local_index] = TransposedCosT[y * 8 + 0]
					* dct_coefficient[start_index + x + 0 * 8]
					+ TransposedCosT[y * 8 + 1] * dct_coefficient[start_index + x + 1 * 8]
					+ TransposedCosT[y * 8 + 2] * dct_coefficient[start_index + x + 2 * 8]
					+ TransposedCosT[y * 8 + 3] * dct_coefficient[start_index + x + 3 * 8]
					+ TransposedCosT[y * 8 + 4] * dct_coefficient[start_index + x + 4 * 8]
					+ TransposedCosT[y * 8 + 5] * dct_coefficient[start_index + x + 5 * 8]
					+ TransposedCosT[y * 8 + 6] * dct_coefficient[start_index + x + 6 * 8]
					+ TransposedCosT[y * 8 + 7] * dct_coefficient[start_index + x + 7 * 8];

				__syncthreads();

				float value = vertical_result[y * 8 + 0] * CosT[x + 0 * 8]
					+ vertical_result[y * 8 + 1] * CosT[x + 1 * 8]
					+ vertical_result[y * 8 + 2] * CosT[x + 2 * 8]
					+ vertical_result[y * 8 + 3] * CosT[x + 3 * 8]
					+ vertical_result[y * 8 + 4] * CosT[x + 4 * 8]
					+ vertical_result[y * 8 + 5] * CosT[x + 5 * 8]
					+ vertical_result[y * 8 + 6] * CosT[x + 6 * 8]
					+ vertical_result[y * 8 + 7] * CosT[x + 7 * 8];

				yuv_result[start_index + local_index] = (byte) ((int) value);
			}

			//-------------------------------------------------------------------------------------------------//
			//
			// 量子化
			//
			//-------------------------------------------------------------------------------------------------//
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
			 *
			 */__global__ void ZigzagQuantizeLow(const int *dct_coefficient, int *quantized, float quariry) {
				using quantize::YUV;
				using Zigzag::sequence;

				int local_index = threadIdx.x + threadIdx.y * 8;
				int start_index = 64 * threadIdx.z + 128 * blockIdx.x + 128 * gridDim.x * blockIdx.y
					+ gridDim.x * 128 * 3 * blockIdx.z;
				quantized[start_index + sequence[local_index]] = dct_coefficient[start_index + local_index]
					/ ((255.0f - ((255.0f - YUV[blockIdx.y][local_index]) * (1.0f + quariry))));
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
			 *
			 */__global__ void InverseZigzagQuantizeLow(const int *quantized, int *dct_coefficient,
				float quariry) {
				using quantize::YUV;
				using Zigzag::sequence;

				int local_index = threadIdx.x + threadIdx.y * 8;
				int start_index = 64 * threadIdx.z + 128 * blockIdx.x + 128 * gridDim.x * blockIdx.y
					+ gridDim.x * 128 * 3 * blockIdx.z;
				dct_coefficient[start_index + local_index] = quantized[start_index + sequence[local_index]]
					* ((255.0f - ((255.0f - YUV[blockIdx.y][local_index]) * (1.0f + quariry))));
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
			 *
			 */__global__ void ZigzagQuantizeHigh(const int *dct_coefficient, int *quantized, float quarity) {
				using quantize::YUV;
				using Zigzag::sequence;

				int local_index = threadIdx.x + threadIdx.y * 8;
				int start_index = 64 * threadIdx.z + 128 * blockIdx.x + 128 * gridDim.x * blockIdx.y
					+ gridDim.x * 128 * 3 * blockIdx.z;
				quantized[start_index + sequence[local_index]] = dct_coefficient[start_index + local_index]
					/ (YUV[blockIdx.y][local_index] * (1.0f - quarity));
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
			 *
			 */__global__ void InverseZigzagQuantizeHigh(const int *quantized, int *dct_coefficient,
				float quarity) {
				using quantize::YUV;
				using Zigzag::sequence;

				int local_index = threadIdx.x + threadIdx.y * 8;
				int start_index = 64 * threadIdx.z + 128 * blockIdx.x + 128 * gridDim.x * blockIdx.y
					+ gridDim.x * 128 * 3 * blockIdx.z;
				dct_coefficient[start_index + local_index] = quantized[start_index + sequence[local_index]]
					* (YUV[blockIdx.y][local_index] * (1.0f - quarity));
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
			 *
			 */__global__ void ZigzagQuantizeMax(const int *dct_coefficient, int *quantized) {
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
			 *
			 */__global__ void InverseZigzagQuantizeMax(const int *quantized, int *dct_coefficient) {
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
			namespace huffman {
				/**
				 * 余ったビットに1を詰めるためのマスク
				 */__device__ __constant__ static const unsigned char kBitFullMaskT[] = {
					0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0x7f, 0xff };
				/**
				 * 余ったビットに1を詰めるためのマスク
				 */__device__ __constant__ static const unsigned char kBitFullMaskLowT[] = {
					0x80, 0xc0, 0xe0, 0xf0, 0xf8, 0xfc, 0xfe, 0xff };

				/**
				 * @brief CUDAによる圧縮のMCUごとのステート
				 *
				 * @author yuumomma
				 * @version 1.0
				 */
				class OutBitStream {
				public:
					int byte_pos_;
					int bit_pos_; 	//! 書き込みビット位置（上位ビットが7、下位ビットが0）
					int writable_; 	//! 1:書き込み可, 0:書き込み不可
					int num_bits_; 	//! 全体バッファに書き込むサイズ

					byte *dst_buf_;

					static const int MAX_BLOCK_SIZE = 128;

					/**
					 * @brief コンストラクタ
					 *
					 * 必ずMAX_BLOCK_SIZE以上のサイズのデバイスメモリを指定すること
					 *
					 * @param device_buf
					 */
					inline __host__ OutBitStream(byte* device_buf) :
						byte_pos_(0),
						bit_pos_(7),
						writable_(1),
						num_bits_(0),
						dst_buf_(device_buf) {
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
				};

				//カーネル呼び出しは8x8ブロックごと=(block_width*block_height*3/2)/64 thread
				//最低は16x16の画像で、8X8ブロックは6個。blockIDx.yでわけるとすると最低スレッド数は1。
				//
				//buffer_size = block_width * block_height * 3 / 2;
				//block_num = buffer_size / 64;
				//grid(block_num / 3 / THREADS, 3, 1)
				//block(THREADS, 1, 1)
				__global__ void HuffmanEncodeForMCU(int *quantized, OutBitStream *dst, int sizeX, int sizeY) {
					using namespace ohmura::encode_table::HuffmanEncode;

					const int mcu_id = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * 3; //マクロブロック番号
					const int mcu_start_index = 64 * mcu_id; // 量子化結果におけるマクロブロックの開始インデックス

					const int Ysize = sizeX * sizeY;
					const int Cbsize = Ysize + sizeX * sizeY / 4;

					OutBitStream *dst_mcu = &dst[mcu_id];

					//Y
					if (mcu_start_index < Ysize) {
						// DC成分
						// DC成分は前のMCUの成分との差分をハフマン符号化するため
						// 画像の最も左上のMCUとの差分は0と取る
						int diff = 0;
						if (mcu_start_index == 0) {
							diff = quantized[mcu_start_index];
						} else {
							diff = quantized[mcu_start_index] - quantized[mcu_start_index - 64];
						}

						// 差分の絶対値から最低限その値を表すのに必要なbitを計算
						// おそらく有効bitは11(=2^11=2048以下)のはず
						byte4 abs_diff = abs(diff);
						int effective_bits = EffectiveBits(abs_diff);
						dst_mcu->setBits(DC::luminance::code[effective_bits],
							DC::luminance::code_size[effective_bits]);

						// ハフマンbitに続けて実値を書き込む
						if (effective_bits != 0) {
							if (diff < 0)
								--diff;
							dst_mcu->setBits(diff, effective_bits);
						}

						// AC成分
						// 残り63のDCT係数に対して処理を行う
						int runlength = 0;
#pragma unroll
						for (int i = 1; i < 64; i++) {
							int value = quantized[mcu_start_index + i];
							byte4 abs_value = abs(value);

							if (abs_value != 0) {
								// 0の個数が16ごとにZRL=code_id:151を代わりに割り当てる
								while (runlength > 15) {
									dst_mcu->setBits(AC::luminance::code[AC::luminance::ZRL],
										AC::luminance::size[AC::luminance::ZRL]);
									runlength -= 16;
								}

								// 有効bit数と15以下の0の個数を合わせて符号化コードを書き込む
								// おそらくAC成分の有効bitは10(=2^10=1024)以下のはず
								// したがってcode_idは[1,150][152,161]の範囲
								effective_bits = EffectiveBits(abs_value);
								int code_id = runlength * 10 + effective_bits + (runlength == 15);
								dst_mcu->setBits(AC::luminance::code[code_id], AC::luminance::size[code_id]);
								runlength = 0;

								// ハフマンbitに続けて実値を書き込む
								if (value < 0)
									--value;
								dst_mcu->setBits(value, effective_bits);

							} else {
								if (i == 63) {
									// 末尾だったときはEOB=code_id:0
									dst_mcu->setBits(AC::luminance::code[AC::luminance::EOB],
										AC::luminance::size[AC::luminance::EOB]);
								} else {
									++runlength;
								}
							}
						}
					}
					// U
					else if (mcu_start_index < Cbsize) {
						// DC成分
						// DC成分は前のMCUの成分との差分をハフマン符号化するため
						// 画像の最も左上のMCUとの差分は0と取る
						int diff = 0;
						if (mcu_start_index == Ysize) {
							diff = quantized[mcu_start_index];
						} else {
							diff = quantized[mcu_start_index] - quantized[mcu_start_index - 64];
						}

						// 差分の絶対値から最低限その値を表すのに必要なbitを計算
						// おそらく有効bitは11(=2^11=2048以下)のはず
						byte4 abs_diff = abs(diff);
						int effective_bits = EffectiveBits(abs_diff);
						dst_mcu->setBits(DC::component::code[effective_bits],
							DC::component::code_size[effective_bits]);

						// ハフマンbitに続けて実値を書き込む
						if (effective_bits) {
							if (diff < 0)
								diff--;
							dst_mcu->setBits(diff, effective_bits);
						}

						// AC成分
						// 残り63のDCT係数に対して処理を行う
						int runlength = 0;
#pragma unroll
						for (int i = 1; i < 64; i++) {
							int value = quantized[mcu_start_index + i];
							byte4 abs_value = abs(value);

							if (abs_value) {
								// 0の個数が16ごとにZRL=code_id:151を代わりに割り当てる
								while (runlength > 15) {
									dst_mcu->setBits(AC::component::code[AC::component::ZRL],
										AC::component::size[AC::component::ZRL]);
									runlength -= 16;
								}

								// 有効bit数と15以下の0の個数を合わせて符号化コードを書き込む
								// おそらくAC成分の有効bitは10(=2^10=1024)以下のはず
								// したがってcode_idは[1,150][152,161]の範囲
								effective_bits = EffectiveBits(abs_value);
								int aIdx = runlength * 10 + effective_bits + (runlength == 15);
								dst_mcu->setBits(AC::component::code[aIdx], AC::component::size[aIdx]);
								runlength = 0;

								// ハフマンbitに続けて実値を書き込む
								if (value < 0)
									--value;
								dst_mcu->setBits(value, effective_bits);

							} else {
								if (i == 63) {
									// 末尾だったときはEOB=code_id:0
									dst_mcu->setBits(AC::component::code[AC::component::EOB],
										AC::component::size[AC::component::EOB]);
								} else
									runlength++;
							}
						}
					}
					// V
					else {
						//DC
						int diff = 0;
						if (mcu_start_index == Cbsize) {
							diff = quantized[mcu_start_index];
						} else
							diff = quantized[mcu_start_index] - quantized[mcu_start_index - 64];
						int absC = abs(diff);
						int dIdx = 0;
						while (absC > 0) {
							absC >>= 1;
							dIdx++;
						}
						dst_mcu->setBits(DC::component::code[dIdx], DC::component::code_size[dIdx]);
						if (dIdx) {
							if (diff < 0)
								diff--;
							dst_mcu->setBits(diff, dIdx);
						}
						int run = 0;

						//AC
						for (int i = 1; i < 64; i++) {
							absC = abs(quantized[mcu_start_index + i]);
							if (absC) {
								while (run > 15) {
									dst_mcu->setBits(AC::component::code[AC::component::ZRL],
										AC::component::size[AC::component::ZRL]);
									run -= 16;
								}
								int s = 0;
								while (absC > 0) {
									absC >>= 1;
									s++;
								}
								int aIdx = run * 10 + s + (run == 15);
								dst_mcu->setBits(AC::component::code[aIdx], AC::component::size[aIdx]);
								int v = quantized[mcu_start_index + i];
								if (v < 0)
									v--;
								dst_mcu->setBits(v, s);
								run = 0;
							} else {
								if (i == 63) {
									dst_mcu->setBits(AC::component::code[AC::component::EOB],
										AC::component::size[AC::component::EOB]);
								} else
									run++;
							}
						}
					}
				}
			}  // namespace huffman

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

	} // namespace cuda
} // namespace jpeg
