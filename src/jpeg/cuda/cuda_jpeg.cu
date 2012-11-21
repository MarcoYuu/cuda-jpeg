#include "stdio.h"
#include "stdlib.h"
#include "cuda_jpeg.cuh"

namespace jpeg {
	namespace cuda {

		using namespace util;
		using namespace util::cuda;

		namespace kernel {
			namespace ver1 {
				__global__ void CreateConversionTable(size_t width, size_t height, size_t block_width,
					size_t block_height, int *table) {
					// ------------------- 各CUDAブロックに対して
					const int grid_col_num = width / block_width;
					const int grid_x = blockIdx.z / grid_col_num;
					const int grid_y = blockIdx.z % grid_col_num;

					// 元画像の各画像ブロックに対する左上インデックス
					const int src_start_index = grid_x * block_width * 3 + grid_y * width * block_height * 3;

					// ブロックごとの書き込み先の先頭アドレス
					const int dst_start_y_index = blockIdx.z * block_width * block_height;

					// ------------------- CUDAブロック内の任意画像ブロック分割に対して
					// 画像ブロック内の16x16ブロックの左上アドレス
					const int src_block_start_index = src_start_index + blockIdx.x * 16 * 3
						+ blockIdx.y * width * 3 * 16;

					// 書き込み先の16x16ブロックごとの先頭アドレス
					const int dst_block_start_y_index = dst_start_y_index + blockIdx.x * 16 * 16
						+ blockIdx.y * 16 * 16 * gridDim.x;

					// ------------------- 16x16 block に関して
					const int x = threadIdx.x;
					const int y = threadIdx.y;
					const int mcu_id_x = x / 8; // 0,1
					const int mcu_id_y = y / 8; // 0,1

					// 元画像の8x8ごとのインデックス
					const int src_8x8_left_up = src_block_start_index + mcu_id_x * 8 * 3 + mcu_id_y * width * 3 * 8;
					const int local_index = x % 8 * 3 + (y % 8) * width * 3;
					const int src_id = src_8x8_left_up + local_index;

					// 書き込み先インデックス
					const int mcu_id = mcu_id_x + mcu_id_y * 2; // 0-4
					const int mcu_offset = mcu_id * 64; // 0, 64, 128, 192
					const int local_dst_index = x % 8 + (y % 8) * 8; // 0-63
					const int dst_id = dst_block_start_y_index + mcu_offset + local_dst_index;

					table[dst_id] = src_id;
				}

				__global__ void ConvertRGBToYUV(const byte* rgb, byte* yuv_result,
					size_t block_size, const int *table) {
					// dst_id = pixel index
					const int dst_id = threadIdx.x + threadIdx.y * blockDim.x
						+ (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y)
							* blockDim.x * blockDim.y;
					const int src_id = table[dst_id];

					const byte b = rgb[src_id + 0] * 0.8588 + 16;
					const byte g = rgb[src_id + 1] * 0.8588 + 16;
					const byte r = rgb[src_id + 2] * 0.8588 + 16;

					// 色変換
					// Y [0,255]
					yuv_result[dst_id] = byte(0.11448 * b + 0.58661 * g + 0.29891 * r);

					//printf("dst_id,%d,yuv_result,%d,r,%d,g,%d,b,%d\n", dst_id, yuv_result[dst_id], r, g, b);

					const int local_dst_c_index = threadIdx.x / 2 + threadIdx.y / 2 * 8; // 0-63
					const int dst_u_id = block_size * blockIdx.z * 3 / 2 + block_size + blockIdx.x * 64
						+ blockIdx.y * 64 * gridDim.x + local_dst_c_index;
					const int dst_v_id = dst_u_id + block_size / 4;

					//printf("pixel,%d,y_id,%d,u_id,%d,v_id,%d\n", src_id / 3, dst_id, dst_u_id, dst_v_id);

					// U,V [-128,127] -> [0,255]
					if (threadIdx.x % 2 == 0 && threadIdx.y % 2 == 0) {
						//yuv_result[dst_u_id] = 0.50000 * b - 0.33126 * g - 0.16874 * r + 128;
						//yuv_result[dst_v_id] = -0.08131 * b - 0.41869 * g + 0.50000 * r + 128;
						//yuv_result[dst_u_id] = 128;
						//yuv_result[dst_v_id] = 128;
					}
				}

				__global__ void ConvertYUVToRGB(const byte* yuv, byte* rgb_result,
					size_t block_size, int *table) {
					const int src_y_id = threadIdx.x + threadIdx.y * blockDim.x
						+ (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y)
							* blockDim.x * blockDim.y;

					const int local_dst_c_index = threadIdx.x / 2 + threadIdx.y / 2 * 8; // 0-63
					const int src_u_id = block_size * blockIdx.z * 3 / 2 + block_size + blockIdx.x * 64
						+ blockIdx.y * 64 * gridDim.x + local_dst_c_index;
					const int src_v_id = src_u_id + block_size / 4;

					const int dst_id = table[src_y_id];

					const int y = yuv[src_y_id];
					//const int u = yuv[src_u_id] - 128;
					//const int v = yuv[src_v_id] - 128;

					//rgb_result[dst_id + 0] = (y + 1.77200 * u - 16) * 1.164;
					//rgb_result[dst_id + 1] = (y - 0.34414 * u - 0.71414 * v - 16) * 1.164;
					//rgb_result[dst_id + 2] = (y + 1.40200 * v - 16) * 1.164;

					rgb_result[dst_id + 0] = y;
					rgb_result[dst_id + 1] = y;
					rgb_result[dst_id + 2] = y;

					//printf("pixel,%d,y_id,%d,u_id,%d,v_id,%d\n", dst_id / 3, src_y_id, src_u_id, src_v_id);
				}
			} // namespace ver1

			namespace ver2 {
				__global__ void CreateConversionTable(size_t width, size_t height, size_t block_width,
					size_t block_height,
					TableElementSrcToDst *table) {

					const size_t img_block_x_num = width / block_width;
					const size_t img_block_size = block_width * block_height;
					const size_t mcu_block_x_num = block_width / 16;

					const size_t block_x = blockIdx.z % img_block_x_num;
					const size_t block_y = blockIdx.z / img_block_x_num;
					const int block_id = block_x + block_y * img_block_x_num;
					const int src_block_start_index = block_y * width * block_height + block_x * block_width;
					const int dst_block_start_y_index = block_id * img_block_size * 3 / 2;
					const size_t dst_block_start_u_index = dst_block_start_y_index + img_block_size;
					const size_t dst_block_start_v_index = dst_block_start_u_index + img_block_size / 4;

					const int mcu_y = blockIdx.x;
					const int mcu_x = blockIdx.y;
					const int mcu_id = mcu_x + mcu_y * mcu_block_x_num;
					const int src_mcu_start_index = src_block_start_index + mcu_y * width * 16 + mcu_x * 16;
					const int dst_mcu_start_y_index = dst_block_start_y_index + mcu_id * 256;
					const size_t dst_mcu_u_start_index = dst_block_start_u_index + mcu_id * 64;
					const size_t dst_mcu_v_start_index = dst_block_start_v_index + mcu_id * 64;

					const int pix_y = threadIdx.x;
					const int pix_x = threadIdx.y;

					const int mcu_id_x = pix_x / 8; // 0,1
					const int mcu_id_y = pix_y / 8; // 0,1
					const int block_8x8_id = mcu_id_x + 2 * mcu_id_y; // 0-3
					const int dst_mcu_y_8x8_index = pix_x % 8 + (pix_y % 8) * 8; // 0-63
					const size_t x = pix_x / 2, y = pix_y / 2; // 0-63

					// RGB画像のピクセルインデックス
					const int src_index = src_mcu_start_index + pix_x + pix_y * width;
					// YUVの書き込みインデックス
					const int dst_y_index = dst_mcu_start_y_index + block_8x8_id * 64 + dst_mcu_y_8x8_index;
					const size_t dst_u_index = dst_mcu_u_start_index + x + y * 8;
					const size_t dst_v_index = dst_mcu_v_start_index + x + y * 8;

					table[src_index].y = dst_y_index;
					table[src_index].u = dst_u_index;
					table[src_index].v = dst_v_index;
				}

				__global__ void ConvertRGBToYUV(const byte* rgb, byte* yuv_result,
					size_t block_size, const TableElementSrcToDst *table) {
					const int pix_index = threadIdx.x + threadIdx.y * blockDim.x
						+ (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y)
							* blockDim.x * blockDim.y;

					const int dst_y_index = table[pix_index].y;
					const int dst_u_index = table[pix_index].u;
					const int dst_v_index = table[pix_index].v;
					const int src_index = pix_index * 3;

					const byte b = rgb[src_index + 0] * 0.8588 + 16;
					const byte g = rgb[src_index + 1] * 0.8588 + 16;
					const byte r = rgb[src_index + 2] * 0.8588 + 16;

					// 色変換
					// Y [0,255]
					yuv_result[dst_y_index] = byte(0.11448 * b + 0.58661 * g + 0.29891 * r);
					yuv_result[dst_u_index] = byte(0.50000 * b - 0.33126 * g - 0.16874 * r + 128);
					yuv_result[dst_v_index] = byte(-0.08131 * b - 0.41869 * g + 0.50000 * r + 128);
				}

				__global__ void ConvertYUVToRGB(const byte* yuv, byte* rgb_result,
					size_t block_size, TableElementSrcToDst *table) {
					const int pix_index = threadIdx.x + threadIdx.y * blockDim.x
						+ (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y)
							* blockDim.x * blockDim.y;

					const int dst_index = pix_index * 3;
					const int src_y_index = table[pix_index].y;
					const int src_u_index = table[pix_index].u;
					const int src_v_index = table[pix_index].v;

					const int y = yuv[src_y_index];
					const int u = yuv[src_u_index] - 128;
					const int v = yuv[src_v_index] - 128;

					rgb_result[dst_index + 0] = byte((y + 1.77200 * u - 16) * 1.164);
					rgb_result[dst_index + 1] = byte((y - 0.34414 * u - 0.71414 * v - 16) * 1.164);
					rgb_result[dst_index + 2] = byte((y + 1.40200 * v - 16) * 1.164);

//					rgb_result[dst_index + 0] = y;
//					rgb_result[dst_index + 1] = y;
//					rgb_result[dst_index + 2] = y;
				}
			} // namespace ver2
		} // namespace kernel

		void CreateConversionTable(size_t width, size_t height, size_t block_width,
			size_t block_height, device_memory<TableElementSrcToDst> &table) {

			const dim3 grid(block_width / 16, block_height / 16,
				width / block_width * height / block_height);
			const dim3 block(16, 16, 1);

			kernel::ver2::CreateConversionTable<<<grid,block>>>(width, height, block_width,
				block_height, table.device_data());
		}

		void ConvertRGBToYUV(const device_memory<byte> &rgb, device_memory<byte> &yuv_result,
			size_t width, size_t height, size_t block_width, size_t block_height,
			const device_memory<TableElementSrcToDst> &table) {

			const dim3 grid(block_width / 16, block_height / 16,
				width / block_width * height / block_height);
			const dim3 block(16, 16, 1);

			kernel::ver2::ConvertRGBToYUV<<<grid,block>>>(
				rgb.device_data(), yuv_result.device_data(),
				block_width*block_height,
				table.device_data());
		}

		void ConvertYUVToRGB(const device_memory<byte> &yuv, device_memory<byte> &rgb_result,
			size_t width, size_t height, size_t block_width, size_t block_height,
			device_memory<TableElementSrcToDst> &table) {

			const dim3 grid(block_width / 16, block_height / 16,
				width / block_width * height / block_height);
			const dim3 block(16, 16, 1);

			kernel::ver2::ConvertYUVToRGB<<<grid,block>>>(
				yuv.device_data(), rgb_result.device_data(),
				block_width*block_height,
				table.device_data());
		}

	/**
	 * pixel番号→Y書き込み位置のマップを作成
	 *
	 * @param width
	 * @param height
	 * @param block_width
	 * @param block_height
	 * @param table
	 */
//		void CreateConversionTable(size_t width, size_t height, size_t block_width, size_t block_height,
//			device_memory<TableElementSrcToDst> &table) {
//
//			const size_t img_block_x_num = width / block_width;
//			const size_t img_block_y_num = height / block_height;
//			const size_t img_block_size = block_width * block_height;
//			const size_t mcu_block_x_num = block_width / 16;
//			const size_t mcu_block_y_num = block_height / 16;
//
//			for (size_t block_y = 0; block_y < img_block_y_num; ++block_y) {
//				for (size_t block_x = 0; block_x < img_block_x_num; ++block_x) {
//					const size_t block_id = block_x + block_y * img_block_x_num;
//					const size_t src_block_start_index = block_y * width * block_height + block_x * block_width;
//					const size_t dst_block_start_y_index = block_id * img_block_size * 3 / 2;
//					const size_t dst_block_start_u_index = dst_block_start_y_index + img_block_size;
//					const size_t dst_block_start_v_index = dst_block_start_u_index + img_block_size / 4;
//
//					for (size_t mcu_y = 0; mcu_y < mcu_block_y_num; ++mcu_y) {
//						for (size_t mcu_x = 0; mcu_x < mcu_block_x_num; ++mcu_x) {
//							const size_t mcu_id = mcu_x + mcu_y * mcu_block_x_num;
//							const size_t src_mcu_start_index = src_block_start_index + mcu_y * width * 16 + mcu_x * 16;
//							const size_t dst_mcu_y_start_index = dst_block_start_y_index + mcu_id * 256;
//							const size_t dst_mcu_u_start_index = dst_block_start_u_index + mcu_id * 64;
//							const size_t dst_mcu_v_start_index = dst_block_start_v_index + mcu_id * 64;
//
//							for (size_t pix_y = 0; pix_y < 16; ++pix_y) {
//								for (size_t pix_x = 0; pix_x < 16; ++pix_x) {
//									const size_t mcu_id_x = pix_x / 8; // 0,1
//									const size_t mcu_id_y = pix_y / 8; // 0,1
//									const size_t block_8x8_id = mcu_id_x + 2 * mcu_id_y; // 0-3
//									const size_t dst_mcu_y_index = pix_x % 8 + (pix_y % 8) * 8; // 0-63
//									const size_t x = pix_x / 2, y = pix_y / 2;
//
//									// RGB画像のピクセルインデックス
//									const size_t src_index = src_mcu_start_index + pix_x + pix_y * width;
//									// YUVの書き込みインデックス
//									const size_t dst_y_index = dst_mcu_y_start_index + block_8x8_id * 64
//										+ dst_mcu_y_index;
//									const size_t dst_u_index = dst_mcu_u_start_index + x + y * 8;
//									const size_t dst_v_index = dst_mcu_v_start_index + x + y * 8;
//
//									table.device_data()[src_index].y = dst_y_index;
//									table.device_data()[src_index].u = dst_u_index;
//									table.device_data()[src_index].v = dst_v_index;
//
//									printf("CreateTableCPU, src_id, %lu, dst_id, %lu\n", src_index,
//										table.device_data()[src_index].y);
//								}
//							}
//						}
//					}
//				}
//			}
//		}
	}// namespace cuda
} // namespace jpeg
