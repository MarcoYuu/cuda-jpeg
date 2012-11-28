#include "stdio.h"
#include "stdlib.h"
#include "cuda_jpeg.cuh"

namespace jpeg {
	namespace cuda {

		using namespace util;
		using namespace util::cuda;
		using util::u_int;

		/**
		 * CUDAカーネル関数
		 */
		namespace kernel {
			/**
			 * DCT用定数
			 */
			namespace DCTConstants {
				/**
				 * 8x8DCTの変換係数行列
				 *
				 */__device__ __constant__ static const float CosT[] = {
					0.35355338, 0.35355338, 0.35355338, 0.35355338, 0.35355338, 0.35355338, 0.35355338, 0.35355338, 0.49039263,
					0.41573480, 0.27778509, 0.09754512, -0.09754516, -0.27778518, -0.41573483, -0.49039266, 0.46193978,
					0.19134171, -0.19134176, -0.46193978, -0.46193978, -0.19134156, 0.19134180, 0.46193978, 0.41573480,
					-0.09754516, -0.49039266, -0.27778500, 0.27778521, 0.49039263, 0.09754504, -0.41573489, 0.35355338,
					-0.35355338, -0.35355332, 0.35355350, 0.35355338, -0.35355362, -0.35355327, 0.35355341, 0.27778509,
					-0.49039266, 0.09754521, 0.41573468, -0.41573489, -0.09754511, 0.49039266, -0.27778542, 0.19134171,
					-0.46193978, 0.46193978, -0.19134195, -0.19134149, 0.46193966, -0.46193987, 0.19134195, 0.09754512,
					-0.27778500, 0.41573468, -0.49039260, 0.49039271, -0.41573480, 0.27778557, -0.09754577 };
				/**
				 * 8x8DCTの変換係数転地行列
				 *
				 */__device__ __constant__ static const float TransposedCosT[] = {
					0.35355338, 0.49039263, 0.46193978, 0.41573480, 0.35355338, 0.27778509, 0.19134171, 0.09754512, 0.35355338,
					0.41573480, 0.19134171, -0.09754516, -0.35355338, -0.49039266, -0.46193978, -0.27778500, 0.35355338,
					0.27778509, -0.19134176, -0.49039266, -0.35355332, 0.09754521, 0.46193978, 0.41573468, 0.35355338,
					0.09754512, -0.46193978, -0.27778500, 0.35355350, 0.41573468, -0.19134195, -0.49039260, 0.35355338,
					-0.09754516, -0.46193978, 0.27778521, 0.35355338, -0.41573489, -0.19134149, 0.49039271, 0.35355338,
					-0.27778518, -0.19134156, 0.49039263, -0.35355362, -0.09754511, 0.46193966, -0.41573480, 0.35355338,
					-0.41573483, 0.19134180, 0.09754504, -0.35355327, 0.49039266, -0.46193987, 0.27778557, 0.35355338,
					-0.49039266, 0.46193978, -0.41573489, 0.35355341, -0.27778542, 0.19134195, -0.09754577 };
			} // namespace DCTConstants

			/**
			 * ジグザグシーケンス用定数
			 */
			namespace Zigzag {
				/**
				 * ジグザグシーケンス用
				 *
				 */__device__ __constant__ static const int sequence[] = {
					0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11, 18, 24, 31,
					40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34, 37, 47, 50, 56, 59, 61,
					35, 36, 48, 49, 57, 58, 62, 63 };
			} // namespace Zigzag

			/**
			 * 量子化テーブル用定数
			 */
			namespace Quantize {
				/**
				 * 輝度用
				 *
				 */__device__ __constant__ static const int luminance[] = {
					16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22,
					29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103,
					121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99 };
				/**
				 * 色差用
				 *
				 */__device__ __constant__ static const int component[] = {
					17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99,
					99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
					99, 99, 99, 99, 99, 99, 99, 99, 99, 99 };

				__device__ __constant__ static const int* YUV[] = { luminance, component };
			} // namespace Quantize

			/**
			 * 色変換テーブル作成カーネル
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
			 */__global__ void CreateConversionTable(u_int width, u_int height, u_int block_width, u_int block_height,
				TableElementSrcToDst *table) {

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
			 * RGBをYUVに変換
			 *
			 * -カーネル起動は各ピクセルごと = width*heightスレッド必要
			 * 例)
			 * 	- grid(block_width/16, block_height/16, width/block_width * height/block_height)
			 * 	- block(16, 16, 1)
			 * -グリッド/ブロック数に制限はない
			 * -rgb.size == yuv.sizeであること
			 *
			 * @param rgb BGRで保存されたソースデータ
			 * @param yuv_result yuvに変換された結果
			 * @param table 変換テーブル
			 *
			 */__global__ void ConvertRGBToYUV(const byte* rgb, byte* yuv_result, const TableElementSrcToDst *table) {
				const int pix_index = threadIdx.x + threadIdx.y * blockDim.x
					+ (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * blockDim.x * blockDim.y;

				const TableElementSrcToDst elem = table[pix_index];
				const u_int src_index = pix_index * 3;

				// R,G,B [0, 255] -> R,G,B [16, 235]
				const float b = rgb[src_index + 0] * 0.8588f + 16.0f;
				const float g = rgb[src_index + 1] * 0.8588f + 16.0f;
				const float r = rgb[src_index + 2] * 0.8588f + 16.0f;

				// R,G,B [16, 235] -> Y [16, 235] U,V [16, 240]
				yuv_result[elem.y] = byte(0.11448f * b + 0.58661f * g + 0.29891f * r);
				yuv_result[elem.u] = byte(0.50000f * b - 0.33126f * g - 0.16874f * r + 128.0f);
				yuv_result[elem.v] = byte(-0.08131f * b - 0.41869f * g + 0.50000f * r + 128.0f);
			}

			/**
			 * YUVをRGBに変換
			 *
			 * -カーネル起動は各ピクセルごと = width*heightスレッド必要
			 * 例)
			 * 	- grid(block_width/16, block_height/16, width/block_width * height/block_height)
			 * 	- block(16, 16, 1)
			 * -グリッド/ブロック数に制限はない
			 * -rgb.size == yuv.sizeであること
			 *
			 * @param yuv YUV411で保存されたソースデータ
			 * @param rgb_result rgbに変換された結果
			 * @param table 変換テーブル
			 *
			 */__global__ void ConvertYUVToRGB(const byte* yuv, byte* rgb_result, const TableElementSrcToDst *table) {
				const int pix_index = threadIdx.x + threadIdx.y * blockDim.x
					+ (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * blockDim.x * blockDim.y;

				const TableElementSrcToDst elem = table[pix_index];
				const u_int dst_index = pix_index * 3;

				// Y [16, 235] U,V [16, 240] -> Y [16, 235] U,V [-112, 112]
				const float y = yuv[elem.y];
				const float u = yuv[elem.u] - 128.0f;
				const float v = yuv[elem.v] - 128.0f;

				// Y [16, 235] U,V [-112, 112] -> R,G,B [0, 255]
				rgb_result[dst_index + 0] = byte((y + 1.77200f * u - 16.0f) * 1.164f);
				rgb_result[dst_index + 1] = byte((y - 0.34414f * u - 0.71414f * v - 16.0f) * 1.164f);
				rgb_result[dst_index + 2] = byte((y + 1.40200f * v - 16.0f) * 1.164f);
			}

			/**
			 * DCTを適用する
			 *
			 * @param yuv 64byte=8x8blockごとに連続したメモリに保存されたデータ
			 * @param dct_coefficient DCT係数
			 *
			 */__global__ void DiscreteCosineTransform(const byte *yuv_src, int *dst_coefficient) {
				using DCTConstants::CosT;
				using DCTConstants::TransposedCosT;

				int x = threadIdx.x, y = threadIdx.y;
				int local_index = x + y * 8;
				int start_index = 64 * blockIdx.x;

				__shared__ float vertical_result[64];

				vertical_result[local_index] = CosT[y * 8 + 0] * yuv_src[start_index + x + 0 * 8]
					+ CosT[y * 8 + 1] * yuv_src[start_index + x + 1 * 8] + CosT[y * 8 + 2] * yuv_src[start_index + x + 2 * 8]
					+ CosT[y * 8 + 3] * yuv_src[start_index + x + 3 * 8] + CosT[y * 8 + 4] * yuv_src[start_index + x + 4 * 8]
					+ CosT[y * 8 + 5] * yuv_src[start_index + x + 5 * 8] + CosT[y * 8 + 6] * yuv_src[start_index + x + 6 * 8]
					+ CosT[y * 8 + 7] * yuv_src[start_index + x + 7 * 8];

				__syncthreads();

				dst_coefficient[start_index + local_index] = vertical_result[y * 8 + 0] * TransposedCosT[x + 0 * 8]
					+ vertical_result[y * 8 + 1] * TransposedCosT[x + 1 * 8]
					+ vertical_result[y * 8 + 2] * TransposedCosT[x + 2 * 8]
					+ vertical_result[y * 8 + 3] * TransposedCosT[x + 3 * 8]
					+ vertical_result[y * 8 + 4] * TransposedCosT[x + 4 * 8]
					+ vertical_result[y * 8 + 5] * TransposedCosT[x + 5 * 8]
					+ vertical_result[y * 8 + 6] * TransposedCosT[x + 6 * 8]
					+ vertical_result[y * 8 + 7] * TransposedCosT[x + 7 * 8];
			}

			/**
			 * iDCTを適用する
			 *
			 * @param dct_coefficient DCT係数
			 * @param yuv_result 64byte=8x8blockごとに連続したメモリに保存されたデータ
			 *
			 */__global__ void InverseDiscreteCosineTransform(const int *dst_coefficient, byte *yuv_result) {
				using DCTConstants::CosT;
				using DCTConstants::TransposedCosT;

				int x = threadIdx.x, y = threadIdx.y;
				int local_index = x + y * 8;
				int start_index = 64 * blockIdx.x;

				__shared__ float vertical_result[64];

				vertical_result[local_index] = TransposedCosT[y * 8 + 0] * dst_coefficient[start_index + x + 0 * 8]
					+ TransposedCosT[y * 8 + 1] * dst_coefficient[start_index + x + 1 * 8]
					+ TransposedCosT[y * 8 + 2] * dst_coefficient[start_index + x + 2 * 8]
					+ TransposedCosT[y * 8 + 3] * dst_coefficient[start_index + x + 3 * 8]
					+ TransposedCosT[y * 8 + 4] * dst_coefficient[start_index + x + 4 * 8]
					+ TransposedCosT[y * 8 + 5] * dst_coefficient[start_index + x + 5 * 8]
					+ TransposedCosT[y * 8 + 6] * dst_coefficient[start_index + x + 6 * 8]
					+ TransposedCosT[y * 8 + 7] * dst_coefficient[start_index + x + 7 * 8];

				__syncthreads();

				float value = vertical_result[y * 8 + 0] * CosT[x + 0 * 8] + vertical_result[y * 8 + 1] * CosT[x + 1 * 8]
					+ vertical_result[y * 8 + 2] * CosT[x + 2 * 8] + vertical_result[y * 8 + 3] * CosT[x + 3 * 8]
					+ vertical_result[y * 8 + 4] * CosT[x + 4 * 8] + vertical_result[y * 8 + 5] * CosT[x + 5 * 8]
					+ vertical_result[y * 8 + 6] * CosT[x + 6 * 8] + vertical_result[y * 8 + 7] * CosT[x + 7 * 8];

				yuv_result[start_index + local_index] = (byte) ((int) value);
			}

			// (dct_coefficient.size() / 64 / 3, 3, 1) (8,8,1)
			__global__ void ZigzagQuantizeLow(const int *dst_coefficient, int *quantized, int quariry) {
				int local_index = threadIdx.x + threadIdx.y * 8;
				int start_index = 64 * (blockIdx.x + blockDim.x * blockIdx.y);
				quantized[start_index + Zigzag::sequence[local_index]] = dst_coefficient[start_index + local_index]
					/ (255 - ((255 - Quantize::YUV[blockIdx.y / 2][local_index]) * (100 + quariry) / 100));
			}

			// (dct_coefficient.size() / 64 / 3, 3, 1) (8,8,1)
			__global__ void ZigzagQuantizeHigh(const int *dst_coefficient, int *quantized, int quariry) {
				int local_index = threadIdx.x + threadIdx.y * 8;
				int start_index = 64 * (blockIdx.x + blockDim.x * blockIdx.y);
				quantized[start_index + Zigzag::sequence[local_index]] = dst_coefficient[start_index + local_index]
					/ Quantize::YUV[blockIdx.y / 2][local_index] * (100 - quariry);
			}

			// (dct_coefficient.size() / 64 / 3, 3, 1) (8,8,1)
			__global__ void InverseZigzagQuantizeLow(const int *dst_quantized, int *coefficient, int quariry) {
				int local_index = threadIdx.x + threadIdx.y * 8;
				int start_index = 64 * (blockIdx.x + blockDim.x * blockIdx.y);
				coefficient[start_index + local_index] = dst_quantized[start_index + Zigzag::sequence[local_index]]
					/ (255 - ((255 - Quantize::YUV[blockIdx.y / 2][Zigzag::sequence[local_index]]) * (100 + quariry) / 100));
			}

			// (dct_coefficient.size() / 64 / 3, 3, 1) (8,8,1)
			__global__ void InverseZigzagQuantizeHigh(const int *dst_quantized, int *coefficient, int quariry) {
				int local_index = threadIdx.x + threadIdx.y * 8;
				int start_index = 64 * (blockIdx.x + blockDim.x * blockIdx.y);
				coefficient[start_index + local_index] = dst_quantized[start_index + Zigzag::sequence[local_index]]
					/ Quantize::YUV[blockIdx.y / 2][Zigzag::sequence[local_index]] * (100 - quariry);
			}
		} // namespace kernel

		void CreateConversionTable(size_t width, size_t height, size_t block_width, size_t block_height, DeviceTable &table) {
			assert(table.size() >= width * height);
			const dim3 grid(block_width / 16, block_height / 16, width / block_width * height / block_height);
			const dim3 block(16, 16, 1);

			kernel::CreateConversionTable<<<grid,block>>>(width, height, block_width,
				block_height, table.device_data());
		}

		void ConvertRGBToYUV(const DeviceByteBuffer &rgb, DeviceByteBuffer &yuv_result, size_t width, size_t height,
			size_t block_width, size_t block_height, const DeviceTable &table) {
			assert(rgb.size()/2 <= yuv_result.size());

			const dim3 grid(block_width / 16, block_height / 16, width / block_width * height / block_height);
			const dim3 block(16, 16, 1);

			kernel::ConvertRGBToYUV<<<grid,block>>>(rgb.device_data(), yuv_result.device_data(), table.device_data());
		}

		void ConvertYUVToRGB(const DeviceByteBuffer &yuv, DeviceByteBuffer &rgb_result, size_t width, size_t height,
			size_t block_width, size_t block_height, const DeviceTable &table) {
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

		void InverseDiscreteCosineTransform(const DeviceIntBuffer &dct_coefficient, DeviceByteBuffer &yuv_result) {
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

		void ZigzagQuantize(const DeviceIntBuffer &dct_coefficient, DeviceIntBuffer &quantized, int quarity) {
			// grid (8x8ブロックの個数, 分割数, 1)
			const dim3 grid(dct_coefficient.size() / 64 / 3, 3, 1);
			// grid (1, 8x8ブロック)
			const dim3 block(8, 8, 1);

			if (quarity < 50) {
				kernel::ZigzagQuantizeLow<<<grid,block>>>(
					dct_coefficient.device_data(), quantized.device_data(), quarity);
			} else {
				kernel::ZigzagQuantizeHigh<<<grid,block>>>(
					dct_coefficient.device_data(), quantized.device_data(), quarity);
			}
		}

		void InverseZigzagQuantize(const DeviceIntBuffer &quantized, DeviceIntBuffer &dct_coefficient, int quarity) {
			// grid (8x8ブロックの個数, 分割数, 1)
			const dim3 grid(quantized.size() / 64 / 3, 3, 1);
			// grid (1, 8x8ブロック)
			const dim3 block(8, 8, 1);

			if (quarity < 50) {
				kernel::InverseZigzagQuantizeLow<<<grid,block>>>(
					quantized.device_data(), dct_coefficient.device_data(), quarity);
			} else {
				kernel::InverseZigzagQuantizeHigh<<<grid,block>>>(
					quantized.device_data(), dct_coefficient.device_data(), quarity);
			}
		}

	} // namespace cuda
} // namespace jpeg
