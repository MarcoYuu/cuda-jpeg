#include "stdio.h"
#include "stdlib.h"
#include "cuda_jpeg.cuh"

namespace jpeg {
	namespace cuda {

		using namespace util;
		using namespace util::cuda;

		__global__ void kernelOfCreateConvertTable(size_t width, size_t height, size_t block_width,
			size_t block_height, int *table) {
			// ------------------- 各CUDAブロックに対して
			const int grid_row_num = height / block_height;
			const int grid_col_num = width / block_width;
			const int grid_x = blockIdx.z / grid_col_num;
			const int grid_y = blockIdx.z % grid_col_num;

			// 元画像の各画像ブロックに対する左上インデックス
			const int src_start_index = grid_y * block_width * 3
				+ grid_x * width * block_height * 3;

			// 書き込み先のブロックごとの先頭アドレス
			const int dst_start_index = blockIdx.z * block_width * block_height * 3 / 2;
			// ブロックごとの書き込み先の先頭アドレス
			const int dst_start_y_index = blockIdx.z * block_width * block_height * 3 / 2;

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
			const int dst_id = dst_block_start_y_index + mcu_offset + local_dst_index;

			table[dst_id] = src_id;
		}

		__global__ void kernelOfConvertRGBToYUV(const byte* rgb, byte* yuv_result,
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

			printf("dst_id,%d,yuv_result,%d,r,%d,g,%d,b,%d\n", dst_id, yuv_result[dst_id], r, g, b);

			const int local_dst_c_index = threadIdx.x / 2 + threadIdx.y / 2 * 8; // 0-63
			const int dst_u_id = block_size * blockIdx.z * 3 / 2 + block_size + blockIdx.x * 64
				+ blockIdx.y * 64 * gridDim.x + local_dst_c_index;
			const int dst_v_id = dst_u_id + block_size / 4;

			//printf("pixel,%d,y_id,%d,u_id,%d,v_id,%d\n", src_id / 3, dst_id, dst_u_id, dst_v_id);

			// U,V [-128,127] -> [0,255]
			if (threadIdx.x % 2 == 0 && threadIdx.y % 2 == 0) {
				yuv_result[dst_u_id] = 0.50000 * b - 0.33126 * g - 0.16874 * r + 128;
				yuv_result[dst_v_id] = -0.08131 * b - 0.41869 * g + 0.50000 * r + 128;
				//yuv_result[dst_u_id] = 128;
				//yuv_result[dst_v_id] = 128;
			}
		}

		__global__ void kernelOfConvertYUVToRGB(const byte* yuv, byte* rgb_result,
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
			const int u = yuv[src_u_id] - 128;
			const int v = yuv[src_v_id] - 128;

			rgb_result[dst_id + 0] = (y + 1.77200 * u - 16) * 1.164;
			rgb_result[dst_id + 1] = (y - 0.34414 * u - 0.71414 * v - 16) * 1.164;
			rgb_result[dst_id + 2] = (y + 1.40200 * v - 16) * 1.164;

			//printf("pixel,%d,y_id,%d,u_id,%d,v_id,%d\n", dst_id / 3, src_y_id, src_u_id, src_v_id);
		}

		void CreateConvertTable(size_t width, size_t height, size_t block_width,
			size_t block_height, device_memory<int> &table) {

			const dim3 grid(block_width / 16, block_height / 16,
				width / block_width * height / block_height);
			const dim3 block(16, 16, 1);

			kernelOfCreateConvertTable<<<grid,block>>>(width, height, block_width,
				block_height, table.device_data());
		}

		void ConvertRGBToYUV(const device_memory<byte> &rgb, device_memory<byte> &yuv_result,
			size_t width, size_t height, size_t block_width, size_t block_height,
			const device_memory<int> &table) {

			const dim3 grid(block_width / 16, block_height / 16,
				width / block_width * height / block_height);
			const dim3 block(16, 16, 1);

			kernelOfConvertRGBToYUV<<<grid,block>>>(
				rgb.device_data(), yuv_result.device_data(),
				block_width*block_height,
				table.device_data());
		}

		void ConvertYUVToRGB(const device_memory<byte> &yuv, device_memory<byte> &rgb_result,
			size_t width, size_t height, size_t block_width, size_t block_height,
			device_memory<int> &table) {

			const dim3 grid(block_width / 16, block_height / 16,
				width / block_width * height / block_height);
			const dim3 block(16, 16, 1);

			kernelOfConvertYUVToRGB<<<grid,block>>>(
				yuv.device_data(), rgb_result.device_data(),
				block_width*block_height,
				table.device_data());
		}}
}
