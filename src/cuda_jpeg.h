/*
 * cuda_jpeg.h
 *
 *  Created on: 2012/11/07
 *      Author: momma
 */

#ifndef CUDA_JPEG_H_
#define CUDA_JPEG_H_

#include "utils/cuda_memory.hpp"

namespace jpeg {
	namespace cuda {

		using namespace util;

		/**
		 * RGBをYUVに変換
		 *
		 * 各ブロックごとに独立したバッファに代入
		 *
		 * - grid(block_width/16, block_height/16, block_num = block_width/16*block_height/16)
		 * - block(16, 16, 1)
		 *
		 * @param rgb
		 * @param yuv_result
		 * @param width
		 * @param height
		 * @param block_width
		 * @param block_heght
		 */__global__ void ConvertRGBToYUV(byte* rgb, byte* yuv_result, size_t width, size_t height, size_t block_width,
			size_t block_height) {

			size_t block_id = blockIdx.x + gridDim.x * blockIdx.y;
			size_t mcu_id = threadIdx.x / 8 + threadIdx.y / 4;
			size_t local_pixel_id = threadIdx.x + blockDim.x * threadIdx.y;

			yuv_result[block_id * 256 + mcu_id * 64 + threadIdx.x % 8 + threadIdx.y % 8 * 8];

			// 各CUDAブロックに対して
			const int block_num = block_width / 16 * block_height / 16;
			for (int grid = 0; grid < block_num; ++grid) {
				const int grid_row_num = height / block_height;
				const int grid_col_num = width / block_width;
				const int grid_x = grid / grid_col_num;
				const int grid_y = grid % grid_col_num;

				// 元画像の各画像ブロックに対する左上インデックス
				const int src_start_index = grid_y * block_width * 3 + grid_x * width * block_height * 3;
				// 書き込み先のブロックごとの先頭アドレス
				const int dst_start_index = grid * block_width * block_height;

				// CUDAブロック内の任意画像ブロック分割に対して
				for (int block_y = 0; block_y < block_height / 16; ++block_y) {
					for (int block_x = 0; block_x < block_width / 16; ++block_x) {
						// 画像ブロック内の16x16ブロックの左上アドレス
						const int src_block_start_index = src_start_index + block_x * 16 * 3 + block_y * width * 3 * 16;

						// 書き込み先の16x16ブロックごとの先頭アドレス
						const int dst_block_start_y_index = dst_start_index + block_x * 16 * 16
							+ block_y * 16 * 16 * block_width / 16;
						const int dst_block_start_u_index = dst_block_start_y_index + block_width * block_height / 4;
						const int dst_block_start_v_index = dst_block_start_u_index + block_width * block_height / 4;

						// 16x16 block に関して
						for (int y = 0; y < 16; ++y) {
							for (int x = 0; x < 16; ++x) {
								const int mcu_id_x = x / 8;
								const int mcu_id_y = y / 8;
								const int mcu_id = mcu_id_x + mcu_id_y * 2; // 0-4
								const int mcu_offset = mcu_id * 64; // 0, 64, 128, 192

								// 元画像の8x8ごとのインデックス
								const int src_8x8_left_up = src_block_start_index + mcu_id_x * 8 * 3
									+ mcu_id_y * width * 3 * 8;
								const int local_index = x % 8 * 3 + (y % 8) * width * 3;
								const int src_id = mcu_offset + local_index;

								// 書き込み先インデックス
								const int local_dst_index = x % 8 + (y % 8) * 8; // 0-63
								const int dst_id = dst_block_start_y_index + local_dst_index;
								const int u_offset = block_width * block_height / 4;
								const int v_offset = block_width * block_height / 2;

								yuv_result[dst_id] = int(
									0.1440f * rgb[src_id] + 0.5870f * rgb[src_id + 1] + 0.2990f * rgb[src_id + 2]
										- 128);

								// TODO 間違ってるからあとで考える
								if (x % 2 == 0 && y % 2 == 0) {
									yuv_result[dst_id + u_offset] = int(
										0.5000f * rgb[src_id] - 0.3313f * rgb[src_id + 1] - 0.1687f * rgb[src_id + 2]); //-128
									yuv_result[dst_id + v_offset] = int(
										-0.0813f * rgb[src_id] - 0.4187f * rgb[src_id + 1] + 0.5000f * rgb[src_id + 2]); //-128
								}

							}
						}
					}
				}
			}
		}
	}
}

#endif /* CUDA_JPEG_H_ */
