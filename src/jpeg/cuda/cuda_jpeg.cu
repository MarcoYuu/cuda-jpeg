#include "stdio.h"
#include "stdlib.h"
#include "cuda_jpeg.cuh"
#include "../ohmura/gpu_jpeg.cuh"

namespace jpeg {
	namespace cuda {

		using namespace util;
		using namespace util::cuda;

		namespace kernel {
			namespace DCTConstants {
				__device__ __constant__ static float kDisSqrt2 = 1.0 / 1.41421356; // 2の平方根の逆数
				__device__ __constant__ static float CosT[] = {
					0.707107, 0.707107, 0.707107, 0.707107, 0.707107, 0.707107, 0.707107, 0.707107, 0.980785, 0.83147, 0.55557,
					0.19509, -0.19509, -0.55557, -0.83147, -0.980785, 0.92388, 0.382683, -0.382683, -0.92388, -0.92388,
					-0.382683, 0.382683, 0.92388, 0.83147, -0.19509, -0.980785, -0.55557, 0.55557, 0.980785, 0.19509, -0.83147,
					0.707107, -0.707107, -0.707107, 0.707107, 0.707107, -0.707107, -0.707107, 0.707107, 0.55557, -0.980785,
					0.19509, 0.83147, -0.83147, -0.19509, 0.980785, -0.55557, 0.382683, -0.92388, 0.92388, -0.382683, -0.382683,
					0.92388, -0.92388, 0.382683, 0.19509, -0.55557, 0.83147, -0.980785, 0.980785, -0.83147, 0.55557, -0.19509 };

				__device__ __constant__ static float ICosT[] = {
					1, 1, 1, 1, 1, 1, 1, 1, 0.980785, 0.83147, 0.55557, 0.19509, -0.19509, -0.55557, -0.83147, -0.980785,
					0.92388, 0.382683, -0.382683, -0.92388, -0.92388, -0.382683, 0.382683, 0.92388, 0.83147, -0.19509,
					-0.980785, -0.55557, 0.55557, 0.980785, 0.19509, -0.83147, 0.707107, -0.707107, -0.707107, 0.707107,
					0.707107, -0.707107, -0.707107, 0.707107, 0.55557, -0.980785, 0.19509, 0.83147, -0.83147, -0.19509,
					0.980785, -0.55557, 0.382683, -0.92388, 0.92388, -0.382683, -0.382683, 0.92388, -0.92388, 0.382683, 0.19509,
					-0.55557, 0.83147, -0.980785, 0.980785, -0.83147, 0.55557, -0.19509 };

				namespace ver2 {
					__device__ __constant__ static const float CosT[] = {
						0.35355338, 0.35355338, 0.35355338, 0.35355338, 0.35355338, 0.35355338, 0.35355338, 0.35355338,
						0.49039263, 0.41573480, 0.27778509, 0.09754512, -0.09754516, -0.27778518, -0.41573483, -0.49039266,
						0.46193978, 0.19134171, -0.19134176, -0.46193978, -0.46193978, -0.19134156, 0.19134180, 0.46193978,
						0.41573480, -0.09754516, -0.49039266, -0.27778500, 0.27778521, 0.49039263, 0.09754504, -0.41573489,
						0.35355338, -0.35355338, -0.35355332, 0.35355350, 0.35355338, -0.35355362, -0.35355327, 0.35355341,
						0.27778509, -0.49039266, 0.09754521, 0.41573468, -0.41573489, -0.09754511, 0.49039266, -0.27778542,
						0.19134171, -0.46193978, 0.46193978, -0.19134195, -0.19134149, 0.46193966, -0.46193987, 0.19134195,
						0.09754512, -0.27778500, 0.41573468, -0.49039260, 0.49039271, -0.41573480, 0.27778557, -0.09754577 };

					__device__ __constant__ static const float TransposedCosT[] = {
						0.35355338, 0.49039263, 0.46193978, 0.41573480, 0.35355338, 0.27778509, 0.19134171, 0.09754512,
						0.35355338, 0.41573480, 0.19134171, -0.09754516, -0.35355338, -0.49039266, -0.46193978, -0.27778500,
						0.35355338, 0.27778509, -0.19134176, -0.49039266, -0.35355332, 0.09754521, 0.46193978, 0.41573468,
						0.35355338, 0.09754512, -0.46193978, -0.27778500, 0.35355350, 0.41573468, -0.19134195, -0.49039260,
						0.35355338, -0.09754516, -0.46193978, 0.27778521, 0.35355338, -0.41573489, -0.19134149, 0.49039271,
						0.35355338, -0.27778518, -0.19134156, 0.49039263, -0.35355362, -0.09754511, 0.46193966, -0.41573480,
						0.35355338, -0.41573483, 0.19134180, 0.09754504, -0.35355327, 0.49039266, -0.46193987, 0.27778557,
						0.35355338, -0.49039266, 0.46193978, -0.41573489, 0.35355341, -0.27778542, 0.19134195, -0.09754577 };
				} // namespace ver2
			} // namespace DCTConstants

			__global__ void CreateConversionTable(size_t width, size_t height, size_t block_width, size_t block_height,
				TableElementSrcToDst *table) {

				using util::u_int;

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

			__global__ void ConvertRGBToYUV(const byte* rgb, byte* yuv_result, size_t block_size,
				const TableElementSrcToDst *table) {
				const int pix_index = threadIdx.x + threadIdx.y * blockDim.x
					+ (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * blockDim.x * blockDim.y;

				const u_int dst_y_index = table[pix_index].y;
				const u_int dst_u_index = table[pix_index].u;
				const u_int dst_v_index = table[pix_index].v;
				const u_int src_index = pix_index * 3;

				const float b = rgb[src_index + 0] * 0.8588f + 16.0f;
				const float g = rgb[src_index + 1] * 0.8588f + 16.0f;
				const float r = rgb[src_index + 2] * 0.8588f + 16.0f;

				// 色変換
				// Y [0,255]
				yuv_result[dst_y_index] = byte(0.11448f * b + 0.58661f * g + 0.29891f * r);
				yuv_result[dst_u_index] = byte(0.50000f * b - 0.33126f * g - 0.16874f * r + 128.0f);
				yuv_result[dst_v_index] = byte(-0.08131f * b - 0.41869f * g + 0.50000f * r + 128.0f);
			}

			__global__ void ConvertYUVToRGB(const byte* yuv, byte* rgb_result, size_t block_size,
				const TableElementSrcToDst *table) {
				const int pix_index = threadIdx.x + threadIdx.y * blockDim.x
					+ (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * blockDim.x * blockDim.y;

				const u_int dst_index = pix_index * 3;
				const u_int src_y_index = table[pix_index].y;
				const u_int src_u_index = table[pix_index].u;
				const u_int src_v_index = table[pix_index].v;

				const float y = yuv[src_y_index];
				const float u = yuv[src_u_index] - 128.0f;
				const float v = yuv[src_v_index] - 128.0f;

				rgb_result[dst_index + 0] = byte((y + 1.77200f * u - 16.0f) * 1.164f);
				rgb_result[dst_index + 1] = byte((y - 0.34414f * u - 0.71414f * v - 16.0f) * 1.164f);
				rgb_result[dst_index + 2] = byte((y + 1.40200f * v - 16.0f) * 1.164f);
			}

			namespace ver1 {
				//各ドットについて一気にDCTを行う。全てグローバルメモリ
				__global__ void DiscreteCosineTransform0(const byte *yuv_src, float *float_result) {
					using DCTConstants::CosT;
					int start_8x8_index = 64 * (blockIdx.x * blockDim.x + threadIdx.x);
					int pix_x = threadIdx.y, pix_y = threadIdx.z;
					float_result[start_8x8_index + pix_x * 8 + pix_y] = yuv_src[start_8x8_index + pix_x * 8 + 0]
						* CosT[pix_y * 8 + 0] + yuv_src[start_8x8_index + pix_x * 8 + 1] * CosT[pix_y * 8 + 1]
						+ yuv_src[start_8x8_index + pix_x * 8 + 2] * CosT[pix_y * 8 + 2]
						+ yuv_src[start_8x8_index + pix_x * 8 + 3] * CosT[pix_y * 8 + 3]
						+ yuv_src[start_8x8_index + pix_x * 8 + 4] * CosT[pix_y * 8 + 4]
						+ yuv_src[start_8x8_index + pix_x * 8 + 5] * CosT[pix_y * 8 + 5]
						+ yuv_src[start_8x8_index + pix_x * 8 + 6] * CosT[pix_y * 8 + 6]
						+ yuv_src[start_8x8_index + pix_x * 8 + 7] * CosT[pix_y * 8 + 7];
				}

				__global__ void DiscreteCosineTransform1(const float *float_result, int *dst_coef) {
					using DCTConstants::CosT;
					int start_8x8_index = 64 * (blockIdx.x * blockDim.x + threadIdx.x);
					int pix_x = threadIdx.y, pix_y = threadIdx.z;
					dst_coef[start_8x8_index + pix_x * 8 + pix_y] = (int) (float_result[start_8x8_index + 0 * 8 + pix_y]
						* CosT[pix_x * 8 + 0] + float_result[start_8x8_index + 1 * 8 + pix_y] * CosT[pix_x * 8 + 1]
						+ float_result[start_8x8_index + 2 * 8 + pix_y] * CosT[pix_x * 8 + 2]
						+ float_result[start_8x8_index + 3 * 8 + pix_y] * CosT[pix_x * 8 + 3]
						+ float_result[start_8x8_index + 4 * 8 + pix_y] * CosT[pix_x * 8 + 4]
						+ float_result[start_8x8_index + 5 * 8 + pix_y] * CosT[pix_x * 8 + 5]
						+ float_result[start_8x8_index + 6 * 8 + pix_y] * CosT[pix_x * 8 + 6]
						+ float_result[start_8x8_index + 7 * 8 + pix_y] * CosT[pix_x * 8 + 7]) / 4;
				}

				//各ドットについて一気にDCTを行う。全てグローバルメモリ
				__global__ void InverseDiscreteCosineTransform0(const int *dst_coef, float *float_result) {
					using DCTConstants::ICosT;
					using DCTConstants::kDisSqrt2;
					int start_8x8_index = 64 * (blockIdx.x * blockDim.x + threadIdx.x);
					int pix_x = threadIdx.y, pix_y = threadIdx.z;
					//uが0~7
					float_result[start_8x8_index + pix_x * 8 + pix_y] = kDisSqrt2 * dst_coef[start_8x8_index + pix_x * 8 + 0]
						* ICosT[0 * 8 + pix_y] + dst_coef[start_8x8_index + pix_x * 8 + 1] * ICosT[1 * 8 + pix_y]
						+ dst_coef[start_8x8_index + pix_x * 8 + 2] * ICosT[2 * 8 + pix_y]
						+ dst_coef[start_8x8_index + pix_x * 8 + 3] * ICosT[3 * 8 + pix_y]
						+ dst_coef[start_8x8_index + pix_x * 8 + 4] * ICosT[4 * 8 + pix_y]
						+ dst_coef[start_8x8_index + pix_x * 8 + 5] * ICosT[5 * 8 + pix_y]
						+ dst_coef[start_8x8_index + pix_x * 8 + 6] * ICosT[6 * 8 + pix_y]
						+ dst_coef[start_8x8_index + pix_x * 8 + 7] * ICosT[7 * 8 + pix_y];
				}

				__global__ void InverseDiscreteCosineTransform1(float *float_result, byte *yuv_result) {
					using DCTConstants::ICosT;
					using DCTConstants::kDisSqrt2;
					int start_8x8_index = 64 * (blockIdx.x * blockDim.x + threadIdx.x);
					int pix_x = threadIdx.y, pix_y = threadIdx.z;
					//vが0~7
					yuv_result[start_8x8_index + pix_x * 8 + pix_y] = (kDisSqrt2 * float_result[start_8x8_index + 0 * 8 + pix_y]
						* ICosT[0 * 8 + pix_x] + float_result[start_8x8_index + 1 * 8 + pix_y] * ICosT[1 * 8 + pix_x]
						+ float_result[start_8x8_index + 2 * 8 + pix_y] * ICosT[2 * 8 + pix_x]
						+ float_result[start_8x8_index + 3 * 8 + pix_y] * ICosT[3 * 8 + pix_x]
						+ float_result[start_8x8_index + 4 * 8 + pix_y] * ICosT[4 * 8 + pix_x]
						+ float_result[start_8x8_index + 5 * 8 + pix_y] * ICosT[5 * 8 + pix_x]
						+ float_result[start_8x8_index + 6 * 8 + pix_y] * ICosT[6 * 8 + pix_x]
						+ float_result[start_8x8_index + 7 * 8 + pix_y] * ICosT[7 * 8 + pix_x]) / 4 + 128;
				}

				__global__ void DiscreteCosineTransformShared(const byte *yuv_src, int *dst_coef) {
					using DCTConstants::CosT;
					int start_8x8_index = 64 * (blockIdx.x * blockDim.x + threadIdx.x);
					int pix_x = threadIdx.y, pix_y = threadIdx.z;

					__shared__ float float_result[64];

					float_result[pix_x * 8 + pix_y] = yuv_src[start_8x8_index + pix_x * 8 + 0] * CosT[pix_y * 8 + 0]
						+ yuv_src[start_8x8_index + pix_x * 8 + 1] * CosT[pix_y * 8 + 1]
						+ yuv_src[start_8x8_index + pix_x * 8 + 2] * CosT[pix_y * 8 + 2]
						+ yuv_src[start_8x8_index + pix_x * 8 + 3] * CosT[pix_y * 8 + 3]
						+ yuv_src[start_8x8_index + pix_x * 8 + 4] * CosT[pix_y * 8 + 4]
						+ yuv_src[start_8x8_index + pix_x * 8 + 5] * CosT[pix_y * 8 + 5]
						+ yuv_src[start_8x8_index + pix_x * 8 + 6] * CosT[pix_y * 8 + 6]
						+ yuv_src[start_8x8_index + pix_x * 8 + 7] * CosT[pix_y * 8 + 7];

					__syncthreads();

					dst_coef[start_8x8_index + pix_x * 8 + pix_y] = (int) (float_result[0 * 8 + pix_y] * CosT[pix_x * 8 + 0]
						+ float_result[1 * 8 + pix_y] * CosT[pix_x * 8 + 1] + float_result[2 * 8 + pix_y] * CosT[pix_x * 8 + 2]
						+ float_result[3 * 8 + pix_y] * CosT[pix_x * 8 + 3] + float_result[4 * 8 + pix_y] * CosT[pix_x * 8 + 4]
						+ float_result[5 * 8 + pix_y] * CosT[pix_x * 8 + 5] + float_result[6 * 8 + pix_y] * CosT[pix_x * 8 + 6]
						+ float_result[7 * 8 + pix_y] * CosT[pix_x * 8 + 7]) / 4;
				}

				__global__ void InverseDiscreteCosineTransformShared(const int *dst_coef, byte *yuv_result) {
					using DCTConstants::ICosT;
					using DCTConstants::kDisSqrt2;

					int start_8x8_index = 64 * (blockIdx.x * blockDim.x + threadIdx.x);
					int pix_x = threadIdx.y, pix_y = threadIdx.z;

					__shared__ float float_result[64];

					//uが0~7
					float_result[pix_x * 8 + pix_y] = kDisSqrt2 * dst_coef[start_8x8_index + pix_x * 8 + 0]
						* ICosT[0 * 8 + pix_y] + dst_coef[start_8x8_index + pix_x * 8 + 1] * ICosT[1 * 8 + pix_y]
						+ dst_coef[start_8x8_index + pix_x * 8 + 2] * ICosT[2 * 8 + pix_y]
						+ dst_coef[start_8x8_index + pix_x * 8 + 3] * ICosT[3 * 8 + pix_y]
						+ dst_coef[start_8x8_index + pix_x * 8 + 4] * ICosT[4 * 8 + pix_y]
						+ dst_coef[start_8x8_index + pix_x * 8 + 5] * ICosT[5 * 8 + pix_y]
						+ dst_coef[start_8x8_index + pix_x * 8 + 6] * ICosT[6 * 8 + pix_y]
						+ dst_coef[start_8x8_index + pix_x * 8 + 7] * ICosT[7 * 8 + pix_y];

					__syncthreads();

					yuv_result[start_8x8_index + pix_x * 8 + pix_y] = (kDisSqrt2 * float_result[start_8x8_index + 0 * 8 + pix_y]
						* ICosT[0 * 8 + pix_x] + float_result[1 * 8 + pix_y] * ICosT[1 * 8 + pix_x]
						+ float_result[2 * 8 + pix_y] * ICosT[2 * 8 + pix_x]
						+ float_result[3 * 8 + pix_y] * ICosT[3 * 8 + pix_x]
						+ float_result[4 * 8 + pix_y] * ICosT[4 * 8 + pix_x]
						+ float_result[5 * 8 + pix_y] * ICosT[5 * 8 + pix_x]
						+ float_result[6 * 8 + pix_y] * ICosT[6 * 8 + pix_x]
						+ float_result[7 * 8 + pix_y] * ICosT[7 * 8 + pix_x]) / 4 + 128;
				}
			} // namespace ver1

			namespace ver2 {
				__global__ void DiscreteCosineTransform(const byte *yuv_src, int *dst_coefficient) {
					using DCTConstants::ver2::CosT;
					using DCTConstants::ver2::TransposedCosT;

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

					dst_coefficient[start_index + local_index] = vertical_result[y * 8 + 0] * TransposedCosT[x + 0 * 8]
						+ vertical_result[y * 8 + 1] * TransposedCosT[x + 1 * 8]
						+ vertical_result[y * 8 + 2] * TransposedCosT[x + 2 * 8]
						+ vertical_result[y * 8 + 3] * TransposedCosT[x + 3 * 8]
						+ vertical_result[y * 8 + 4] * TransposedCosT[x + 4 * 8]
						+ vertical_result[y * 8 + 5] * TransposedCosT[x + 5 * 8]
						+ vertical_result[y * 8 + 6] * TransposedCosT[x + 6 * 8]
						+ vertical_result[y * 8 + 7] * TransposedCosT[x + 7 * 8];
				}

				__global__ void InverseDiscreteCosineTransform(const int *dst_coefficient, byte *yuv_result) {
					using DCTConstants::ver2::CosT;
					using DCTConstants::ver2::TransposedCosT;

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

					printf("v, %d, %d, %8f\n", blockIdx.x, local_index, vertical_result[local_index]);

					float value = vertical_result[y * 8 + 0] * CosT[x + 0 * 8] + vertical_result[y * 8 + 1] * CosT[x + 1 * 8]
						+ vertical_result[y * 8 + 2] * CosT[x + 2 * 8] + vertical_result[y * 8 + 3] * CosT[x + 3 * 8]
						+ vertical_result[y * 8 + 4] * CosT[x + 4 * 8] + vertical_result[y * 8 + 5] * CosT[x + 5 * 8]
						+ vertical_result[y * 8 + 6] * CosT[x + 6 * 8] + vertical_result[y * 8 + 7] * CosT[x + 7 * 8];

					yuv_result[start_index + local_index] = (byte) ((int) value);

					printf("res, %d, %d, %8f\n", blockIdx.x, start_index + local_index, value);
				}
			} // namespace ver2
		} // namespace kernel

		void CreateConversionTable(size_t width, size_t height, size_t block_width, size_t block_height, DeviceTable &table) {

			const dim3 grid(block_width / 16, block_height / 16, width / block_width * height / block_height);
			const dim3 block(16, 16, 1);

			kernel::CreateConversionTable<<<grid,block>>>(width, height, block_width,
				block_height, table.device_data());
		}

		void ConvertRGBToYUV(const DeviceByteBuffer &rgb, DeviceByteBuffer &yuv_result, size_t width, size_t height,
			size_t block_width, size_t block_height, const DeviceTable &table) {

			const dim3 grid(block_width / 16, block_height / 16, width / block_width * height / block_height);
			const dim3 block(16, 16, 1);

			kernel::ConvertRGBToYUV<<<grid,block>>>(
				rgb.device_data(), yuv_result.device_data(),
				block_width*block_height,
				table.device_data());
		}

		void ConvertYUVToRGB(const DeviceByteBuffer &yuv, DeviceByteBuffer &rgb_result, size_t width, size_t height,
			size_t block_width, size_t block_height, const DeviceTable &table) {

			const dim3 grid(block_width / 16, block_height / 16, width / block_width * height / block_height);
			const dim3 block(16, 16, 1);

			kernel::ConvertYUVToRGB<<<grid,block>>>(
				yuv.device_data(), rgb_result.device_data(),
				block_width*block_height,
				table.device_data());
		}

		void DiscreteCosineTransform(const DeviceByteBuffer &yuv, DeviceIntBuffer &dct_coefficient, size_t width, size_t height,
			size_t block_width, size_t block_height) {
			// grid (8x8ブロックの個数, 分割数, 1)
			const dim3 grid(yuv.size() / 64, 1, 1);
			// grid (1, 8x8ブロック)
			const dim3 block(8, 8, 1);

			kernel::ver2::DiscreteCosineTransform<<<grid,block>>>(
				yuv.device_data(), dct_coefficient.device_data());
		}

		void InverseDiscreteCosineTransform(const DeviceIntBuffer &dct_coefficient, DeviceByteBuffer &yuv_result, size_t width,
			size_t height, size_t block_width, size_t block_height) {
			// grid (8x8ブロックの個数, 分割数, 1)
			const dim3 grid(dct_coefficient.size() / 64, 1, 1);
			// grid (1, 8x8ブロック)
			const dim3 block(8, 8, 1);

			kernel::ver2::InverseDiscreteCosineTransform<<<grid,block>>>(
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
	} // namespace cuda
} // namespace jpeg
