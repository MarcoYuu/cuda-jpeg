#include "stdio.h"
#include "stdlib.h"
#include "cuda_jpeg.cuh"

namespace jpeg {
	namespace cuda {

		using namespace util;

#ifdef DEBUG
		__global__ void ConvertRGBToYUV(
			const byte* rgb,
			byte* yuv_result,
			size_t width,
			size_t height,
			size_t block_width,
			size_t block_height, int *result) {
#else
		__global__ void ConvertRGBToYUV(
			const byte* rgb,
			byte* yuv_result,
			size_t width,
			size_t height,
			size_t block_width,
			size_t block_height) {
#endif
			// ------------------- 各CUDAブロックに対して
			const int grid_row_num = height / block_height;
			const int grid_col_num = width / block_width;
			const int grid_x = blockIdx.z / grid_col_num;
			const int grid_y = blockIdx.z % grid_col_num;

			// 元画像の各画像ブロックに対する左上インデックス
			const int src_start_index = grid_y * block_width * 3 + grid_x * width * block_height * 3;

			// 書き込み先のブロックごとの先頭アドレス
			const int dst_start_index = blockIdx.z * block_width * block_height * 3 / 2;

			// ------------------- CUDAブロック内の任意画像ブロック分割に対して
			// 画像ブロック内の16x16ブロックの左上アドレス
			const int src_block_start_index = src_start_index + blockIdx.x * 16 * 3 + blockIdx.y * width * 3 * 16;

			// 書き込み先の16x16ブロックごとの先頭アドレス
			const int dst_block_start_y_index = dst_start_index + blockIdx.x * 16 * 16
				+ blockIdx.y * 16 * 16 * block_width / 16;
			const int dst_block_start_u_index = dst_block_start_y_index + block_width * block_height / 4;
			const int dst_block_start_v_index = dst_block_start_u_index + block_width * block_height / 4;

			// ------------------- 16x16 block に関して
			const int x = threadIdx.x;
			const int y = threadIdx.y;
			const int mcu_id_x = x / 8;
			const int mcu_id_y = y / 8;
			const int mcu_id = mcu_id_x + mcu_id_y * 2; // 0-4
			const int mcu_offset = mcu_id * 64; // 0, 64, 128, 192

			// 元画像の8x8ごとのインデックス
			const int src_8x8_left_up = src_block_start_index + mcu_id_x * 8 * 3 + mcu_id_y * width * 3 * 8;
			const int local_index = x % 8 * 3 + (y % 8) * width * 3;
			const int src_id = mcu_offset + local_index;

			// 書き込み先インデックス
			const int local_dst_index = x % 8 + (y % 8) * 8; // 0-63
			const int dst_id = dst_block_start_y_index + mcu_offset + local_dst_index;

#ifdef DEBUG
			result[dst_id] = src_id;
#endif

			// 色変換
			//Y [0,255]
			yuv_result[dst_id] = byte(
				0.1440 * rgb[src_id] + 0.5870 * rgb[src_id + 1] + 0.29990 * rgb[src_id + 2]);

			printf("%d, %d\n", src_id, rgb[src_id + 2]);

			//U,V [-128,127] -> [0,255]
			if (x % 2 == 0 && y % 2 == 0) {
				const int local_dst_c_index = x / 2 + y / 8 * 8; // 0-63
				const int dst_u_id = dst_block_start_u_index + local_dst_c_index;
				const int dst_v_id = dst_block_start_v_index + local_dst_c_index;
				yuv_result[dst_u_id] = byte(
					0.5000 * rgb[src_id] - 0.3313 * rgb[src_id + 1] - 0.1687 * rgb[src_id + 2] + 128);
				yuv_result[dst_v_id] = byte(
					-0.0813 * rgb[src_id] - 0.4187 * rgb[src_id + 1] + 0.5000 * rgb[src_id + 2] + 128);
			}
		}

		__global__ void ConvertYUVToRGB(const byte* yuv, byte* rgb_result, size_t width, size_t height,
			size_t block_width, size_t block_height) {

			// ------------------- 各CUDAブロックに対して
			const int grid_col_num = width / block_width;
			const int grid_x = blockIdx.z / grid_col_num;
			const int grid_y = blockIdx.z % grid_col_num;

			const int block_byte_size = block_width * block_height * 3 / 2;

			// 元画像の各画像ブロックに対する左上インデックス
			const int src_start_index = grid_x * block_byte_size + grid_y * block_byte_size * grid_col_num;

			// ------------------- CUDAブロック内の任意画像ブロック分割に対して
			// 画像ブロック内の16x16ブロックの左上アドレス
			const int src_block_start_y_index = src_start_index + blockIdx.x * 256 + blockIdx.y * blockDim.x * 256;
			const int src_block_start_u_index = src_block_start_y_index + block_width * block_height / 4
				+ blockIdx.x * 64
				+ blockIdx.y * blockDim.x * 64;
			const int src_block_start_v_index = src_block_start_y_index + block_width * block_height / 2
				+ blockIdx.x * 64
				+ blockIdx.y * blockDim.x * 64;

			// ------------------- 16x16 block に関して
			const int x = threadIdx.x;
			const int y = threadIdx.y;
			const int mcu_id_x = x / 8; // 0,1
			const int mcu_id_y = y / 8; // 0,1
			const int mcu_id = mcu_id_x + mcu_id_y * 2; // 0-4
			const int mcu_offset = mcu_id * 64; // 0, 64, 128, 192
			const int local_src_index = x % 8 + (y % 8) * 8; // 0-63
			const int src_id = src_block_start_y_index + local_src_index;
		}
	}
}
