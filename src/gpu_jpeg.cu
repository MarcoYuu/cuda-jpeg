#include <cstdio>
#include <cstdlib>

#include "utils/util_cv.h"
#include "utils/encoder_tables_device.cuh"
#include "utils/out_bit_stream.h"
#include "utils/in_bit_stream.h"

#include "utils/gpu_out_bit_stream.cuh"

#include "gpu_jpeg.cuh"
#include "cpu_jpeg.h"

namespace jpeg {
	namespace cuda {
		using namespace encode_table;

		__device__ __constant__ static float kDisSqrt2 = 1.0 / 1.41421356; // 2の平方根の逆数

		__device__ __constant__ static float CosT[] = {
			0.707107, 0.707107, 0.707107, 0.707107, 0.707107, 0.707107, 0.707107, 0.707107, 0.980785,
			0.83147, 0.55557, 0.19509, -0.19509, -0.55557, -0.83147, -0.980785, 0.92388, 0.382683,
			-0.382683, -0.92388, -0.92388, -0.382683, 0.382683, 0.92388, 0.83147, -0.19509, -0.980785,
			-0.55557, 0.55557, 0.980785, 0.19509, -0.83147, 0.707107, -0.707107, -0.707107, 0.707107,
			0.707107, -0.707107, -0.707107, 0.707107, 0.55557, -0.980785, 0.19509, 0.83147, -0.83147,
			-0.19509, 0.980785, -0.55557, 0.382683, -0.92388, 0.92388, -0.382683, -0.382683, 0.92388,
			-0.92388, 0.382683, 0.19509, -0.55557, 0.83147, -0.980785, 0.980785, -0.83147, 0.55557,
			-0.19509
		};

		__device__ __constant__ static float ICosT[] = {
			1, 1, 1, 1, 1, 1, 1, 1, 0.980785, 0.83147, 0.55557, 0.19509, -0.19509, -0.55557, -0.83147,
			-0.980785, 0.92388, 0.382683, -0.382683, -0.92388, -0.92388, -0.382683, 0.382683, 0.92388,
			0.83147, -0.19509, -0.980785, -0.55557, 0.55557, 0.980785, 0.19509, -0.83147, 0.707107,
			-0.707107, -0.707107, 0.707107, 0.707107, -0.707107, -0.707107, 0.707107, 0.55557, -0.980785,
			0.19509, 0.83147, -0.83147, -0.19509, 0.980785, -0.55557, 0.382683, -0.92388, 0.92388,
			-0.382683, -0.382683, 0.92388, -0.92388, 0.382683, 0.19509, -0.55557, 0.83147, -0.980785,
			0.980785, -0.83147, 0.55557, -0.19509
		};

		__device__ byte revise_value_d(double v) {
			if (v < 0.0)
				return 0;
			if (v > 255.0)
				return 255;
			return (byte) v;
		}

		//-------------------------------------------------------------------
		// テーブル作成
		//===================================================================
		void make_trans_table(int *trans_table_Y, int *trans_table_C, int sizeX, int sizeY);
		void make_itrans_table(int *itrans_table_Y, int *itrans_table_C, int sizeX, int sizeY);

		//-------------------------------------------------------------------
		// kernel コード
		//
		//===================================================================
		//-------------------------------------------------------------------
		// color conversion
		//-------------------------------------------------------------------
		//コンスタントメモリ使うと速くなるかも
		__global__ void gpu_color_trans_Y(unsigned char *src_img, int *dst_img, int *trans_table_Y);
		__global__ void gpu_color_trans_C(unsigned char *src_img, int *dst_img, int *trans_table_C,
			const int sizeY, const int C_size);
		__global__ void gpu_color_itrans(int *src_img, unsigned char *dst_img, int *itrans_table_Y,
			int *itrans_table_C, int C_size);

		//-------------------------------------------------------------------
		// DCT
		//-------------------------------------------------------------------
		//各ドットについて一気にDCTを行う。全てグローバルメモリ
		__global__ void gpu_dct_0(int *src_ycc, float *pro_f);
		__global__ void gpu_dct_1(float *pro_f, int *dst_coef);

		//-------------------------------------------------------------------
		// Inverce-DCT
		//-------------------------------------------------------------------
		//各ドットについて一気にDCTを行う。全てグローバルメモリ
		__global__ void gpu_idct_0(int *src_ycc, float *pro_f);
		__global__ void gpu_idct_1(float *pro_f, int *dst_coef);

		//-------------------------------------------------------------------
		// Zig-Zag Quantization
		//-------------------------------------------------------------------
		__global__ void gpu_zig_quantize_Y(int *src_coef, int *dst_qua);
		__global__ void gpu_zig_quantize_C(int *src_coef, int *dst_qua, int size);
		__global__ void gpu_izig_quantize_Y(int *src_qua, int *dst_coef);
		__global__ void gpu_izig_quantize_C(int *src_qua, int *dst_coef, int size);

		//-------------------------------------------------------------------
		// Huffman Coding
		//-------------------------------------------------------------------
		__global__ void gpu_huffman_mcu(int *src_qua, jpeg::cuda::GPUOutBitStreamState *mOBSP, byte *mBufP,
			byte *mEndOfBufP, int sizeX, int sizeY);

		//完全逐次処理、CPUで行った方が圧倒的に速い
		void cpu_huffman_middle(jpeg::cuda::GPUOutBitStreamState *ImOBSP, int sizeX, int sizeY, byte* dst_NumBits);

		//排他処理のため3つに分ける
		//1MCUは最小4bit(EOBのみ)なので1Byteのバッファに最大3MCUが競合する。だから3つに分ける。
		__global__ void gpu_huffman_write_devide0(jpeg::cuda::GPUOutBitStreamState *mOBSP, byte *mBufP, byte *OmBufP,
			int sizeX, int sizeY);
		__global__ void gpu_huffman_write_devide1(jpeg::cuda::GPUOutBitStreamState *mOBSP, byte *mBufP, byte *OmBufP,
			int sizeX, int sizeY);
		__global__ void gpu_huffman_write_devide2(jpeg::cuda::GPUOutBitStreamState *mOBSP, byte *mBufP, byte *OmBufP,
			int sizeX, int sizeY);

		void make_trans_table(int *trans_table_Y, int *trans_table_C, int sizeX, int sizeY) {
			int i, j, k, l, m;
			int src_offset[4] = { 0, 8, 8 * sizeX, 8 * sizeX + 8 };
			int dst_offset[4] = { 0, 64, 128, 192 };
			//int src_posi,dst_posi;
			int MCU_x = sizeX / 16, MCU_y = sizeY / 16;
			//Y
			for (j = 0; j < MCU_y; j++) {
				for (i = 0; i < MCU_x; i++) {
					for (k = 0; k < 4; k++) {
						for (l = 0; l < 8; l++) { //tate
							for (m = 0; m < 8; m++) { //yoko
								trans_table_Y[16 * i + 16 * sizeX * j + src_offset[k] + l * sizeX + m] = 256
									* (i + j * MCU_x) + dst_offset[k] + 8 * l + m;
							}
						}
					}
				}
			}
			//CC
			//cpu用

			for (j = 0; j < MCU_y; j++) {
				for (i = 0; i < MCU_x; i++) {
					for (l = 0; l < 16; l += 2) { //tate
						for (m = 0; m < 16; m += 2) { //yoko
							trans_table_C[16 * i + 16 * sizeX * j + l * sizeX + m] = (sizeX * sizeY)
								+ 64 * (i + j * MCU_x) + 8 * (l / 2) + m / 2;
						}
					}
				}
			}
		}

		void make_itrans_table(int *itrans_table_Y, int *itrans_table_C, int sizeX, int sizeY) {
			int i, j, k, l, m;
			int src_offset[4] = { 0, 64, 128, 192 };
			int dst_offset[4] = { 0, 8, 8 * sizeX, 8 * sizeX + 8 };

			int MCU_x = sizeX / 16, MCU_y = sizeY / 16;
			int Y_size = sizeX * sizeY;
			//Y
			for (j = 0; j < MCU_y; j++) {
				for (i = 0; i < MCU_x; i++) {
					for (k = 0; k < 4; k++) {
						for (l = 0; l < 8; l++) { //tate
							for (m = 0; m < 8; m++) { //yoko
								itrans_table_Y[256 * (i + j * MCU_x) + src_offset[k] + 8 * l + m] = 3
									* (16 * i + 16 * sizeX * j + dst_offset[k] + sizeX * l + m);
								itrans_table_C[256 * (i + j * MCU_x) + src_offset[k] + 8 * l + m] = Y_size
									+ 64 * (i + j * MCU_x) + Sampling::luminance[src_offset[k] + 8 * l + m];
							}
						}
					}
				}
			}
		}

		//コンスタントメモリ使うと速くなるかも
		__global__ void gpu_color_trans_Y(unsigned char *src_img, int *dst_img, int *trans_table_Y) {
			int id = blockIdx.x * blockDim.x + threadIdx.x;
			dst_img[trans_table_Y[id]] = int(
				0.1440 * src_img[3 * id] + 0.5870 * src_img[3 * id + 1] + 0.2990 * src_img[3 * id + 2]
					- 128); //-128
		}

		//2はサイズに関係なくできる。速度はあんま変わらなかった
		__global__ void gpu_color_trans_C(unsigned char *src_img, int *dst_img, int *trans_table_C,
			const int sizeY, const int C_size) {
			//__syncthreads();
			int id = blockIdx.x * blockDim.x + threadIdx.x;
			if (id % 2 == 0 && (id / sizeY) % 2 == 0) {
				dst_img[trans_table_C[id]] = int(
					0.5000 * src_img[3 * id] - 0.3313 * src_img[3 * id + 1] - 0.1687 * src_img[3 * id + 2]); //-128
				dst_img[trans_table_C[id] + C_size] = int(
					-0.0813 * src_img[3 * id] - 0.4187 * src_img[3 * id + 1]
						+ 0.5000 * src_img[3 * id + 2]); //-128
			}
		}

		__global__ void gpu_color_itrans(int *src_img, unsigned char *dst_img, int *itrans_table_Y,
			int *itrans_table_C, int C_size) {
			int id = blockIdx.x * blockDim.x + threadIdx.x;
			dst_img[itrans_table_Y[id]] = revise_value_d(
				src_img[id] + 1.77200 * (src_img[itrans_table_C[id]] - 128));
			dst_img[itrans_table_Y[id] + 1] = revise_value_d(
				src_img[id] - 0.34414 * (src_img[itrans_table_C[id]] - 128)
					- 0.71414 * (src_img[itrans_table_C[id] + C_size] - 128));
			dst_img[itrans_table_Y[id] + 2] = revise_value_d(
				src_img[id] + 1.40200 * (src_img[itrans_table_C[id] + C_size] - 128));
		}

		//各ドットについて一気にDCTを行う。全てグローバルメモリ
		__global__ void gpu_dct_0(int *src_ycc, float *pro_f) {
			int id = 64 * (blockIdx.x * (blockDim.x) + threadIdx.x);
			int y = threadIdx.y, u = threadIdx.z;
			pro_f[id + y * 8 + u] = src_ycc[id + y * 8 + 0] * CosT[u * 8 + 0]
				+ src_ycc[id + y * 8 + 1] * CosT[u * 8 + 1] + src_ycc[id + y * 8 + 2] * CosT[u * 8 + 2]
				+ src_ycc[id + y * 8 + 3] * CosT[u * 8 + 3] + src_ycc[id + y * 8 + 4] * CosT[u * 8 + 4]
				+ src_ycc[id + y * 8 + 5] * CosT[u * 8 + 5] + src_ycc[id + y * 8 + 6] * CosT[u * 8 + 6]
				+ src_ycc[id + y * 8 + 7] * CosT[u * 8 + 7];
		}
		__global__ void gpu_dct_1(float *pro_f, int *dst_coef) {
			int id = 64 * (blockIdx.x * (blockDim.x) + threadIdx.x);
			int v = threadIdx.y, u = threadIdx.z;
			dst_coef[id + v * 8 + u] = int(
				(pro_f[id + 0 * 8 + u] * CosT[v * 8 + 0] + pro_f[id + 1 * 8 + u] * CosT[v * 8 + 1]
					+ pro_f[id + 2 * 8 + u] * CosT[v * 8 + 2] + pro_f[id + 3 * 8 + u] * CosT[v * 8 + 3]
					+ pro_f[id + 4 * 8 + u] * CosT[v * 8 + 4] + pro_f[id + 5 * 8 + u] * CosT[v * 8 + 5]
					+ pro_f[id + 6 * 8 + u] * CosT[v * 8 + 6] + pro_f[id + 7 * 8 + u] * CosT[v * 8 + 7])
					/ 4);
		}

		//各ドットについて一気にDCTを行う。全てグローバルメモリ
		__global__ void gpu_idct_0(int *src_ycc, float *pro_f) {
			int id = 64 * (blockIdx.x * (blockDim.x) + threadIdx.x);
			int v = threadIdx.y, x = threadIdx.z;
			//uが0~7
			pro_f[id + v * 8 + x] = kDisSqrt2 * src_ycc[id + v * 8 + 0] * ICosT[0 * 8 + x] //kDisSqrt2 = Cu
			+ src_ycc[id + v * 8 + 1] * ICosT[1 * 8 + x] + src_ycc[id + v * 8 + 2] * ICosT[2 * 8 + x]
				+ src_ycc[id + v * 8 + 3] * ICosT[3 * 8 + x] + src_ycc[id + v * 8 + 4] * ICosT[4 * 8 + x]
				+ src_ycc[id + v * 8 + 5] * ICosT[5 * 8 + x] + src_ycc[id + v * 8 + 6] * ICosT[6 * 8 + x]
				+ src_ycc[id + v * 8 + 7] * ICosT[7 * 8 + x];
		}
		__global__ void gpu_idct_1(float *pro_f, int *dst_coef) {
			int id = 64 * (blockIdx.x * (blockDim.x) + threadIdx.x);
			int y = threadIdx.y, x = threadIdx.z;
			//vが0~7
			dst_coef[id + y * 8 + x] = int(
				(kDisSqrt2 * pro_f[id + 0 * 8 + x] * ICosT[0 * 8 + y] //kDisSqrt2 = Cv
				+ pro_f[id + 1 * 8 + x] * ICosT[1 * 8 + y] + pro_f[id + 2 * 8 + x] * ICosT[2 * 8 + y]
					+ pro_f[id + 3 * 8 + x] * ICosT[3 * 8 + y] + pro_f[id + 4 * 8 + x] * ICosT[4 * 8 + y]
					+ pro_f[id + 5 * 8 + x] * ICosT[5 * 8 + y] + pro_f[id + 6 * 8 + x] * ICosT[6 * 8 + y]
					+ pro_f[id + 7 * 8 + x] * ICosT[7 * 8 + y]) / 4 + 128);
		}

		__global__ void gpu_zig_quantize_Y(int *src_coef, int *dst_qua) {
			int id = blockIdx.x * blockDim.x + threadIdx.x;
			dst_qua[64 * (id / 64) + Zigzag::sequence[id % 64]] = src_coef[id]
				/ Quantize::luminance[id % 64];
		}

		__global__ void gpu_zig_quantize_C(int *src_coef, int *dst_qua, int size) {
			int id = blockIdx.x * blockDim.x + threadIdx.x + size;
			dst_qua[64 * (id / 64) + Zigzag::sequence[id % 64]] = src_coef[id]
				/ Quantize::component[id % 64];
		}

		__global__ void gpu_izig_quantize_Y(int *src_qua, int *dst_coef) {
			int id = blockIdx.x * blockDim.x + threadIdx.x;
			dst_coef[id] = src_qua[64 * (id / 64) + Zigzag::sequence[id % 64]]
				* Quantize::luminance[id % 64];
		}

		__global__ void gpu_izig_quantize_C(int *src_qua, int *dst_coef, int size) {

			int id = blockIdx.x * blockDim.x + threadIdx.x + size;
			dst_coef[id] = src_qua[64 * (id / 64) + Zigzag::sequence[id % 64]]
				* Quantize::component[id % 64];

		}

		__global__ void gpu_huffman_mcu(int *src_qua, GPUOutBitStreamState *dst, byte *mBufP,
			byte *mEndOfBufP, int sizeX, int sizeY) { //,
			int id = blockIdx.x * blockDim.x + threadIdx.x; //マクロブロック番号
			int mid = 64 * (blockIdx.x * blockDim.x + threadIdx.x);

			int i, s, v;
			int diff;
			int absC;
			int dIdx, aIdx;
			const int Ysize = sizeX * sizeY;
			const int Cbsize = Ysize + sizeX * sizeY / 4;

			using namespace HuffmanEncode;

			int run = 0;
			byte *tmp_p = &mBufP[MBS * id];
			//Y
			if (mid < Ysize) {
				//先にDCやらないと・・・
				//DC
				if (mid == 0) {
					diff = src_qua[mid];
				} else {
					diff = src_qua[mid] - src_qua[mid - 64];
				}
				absC = abs(diff);
				dIdx = 0;
				while (absC > 0) {
					absC >>= 1;
					dIdx++;
				}
				SetBits(&dst[id], tmp_p, mEndOfBufP, DC::luminance::code[dIdx], DC::luminance::size[dIdx]);
				if (dIdx) {
					if (diff < 0)
						diff--;
					SetBits(&dst[id], tmp_p, mEndOfBufP, diff, dIdx);
				}
				run = 0;

				//AC
				for (i = 1; i < 64; i++) {
					absC = abs(src_qua[mid + i]);
					if (absC) {
						while (run > 15) {
							SetBits(&dst[id], tmp_p, mEndOfBufP, AC::luminance::code[AC::luminance::ZRL],
								AC::luminance::size[AC::luminance::ZRL]);
							run -= 16;
						}
						s = 0;
						while (absC > 0) {
							absC >>= 1;
							s++;
						}
						aIdx = run * 10 + s + (run == 15);
						SetBits(&dst[id], tmp_p, mEndOfBufP, AC::luminance::code[aIdx],
							AC::luminance::size[aIdx]);
						v = src_qua[mid + i];
						if (v < 0)
							v--;
						SetBits(&dst[id], tmp_p, mEndOfBufP, v, s);
						run = 0;
					} else {
						if (i == 63) {
							SetBits(&dst[id], tmp_p, mEndOfBufP, AC::luminance::code[AC::luminance::EOB],
								AC::luminance::size[AC::luminance::EOB]);
						} else
							run++;
					}
				}
			} else if (mid < Cbsize) {
				//DC
				if (mid == Ysize) {
					diff = src_qua[mid];
				} else
					diff = src_qua[mid] - src_qua[mid - 64];
				absC = abs(diff);
				dIdx = 0;
				while (absC > 0) {
					absC >>= 1;
					dIdx++;
				}
				SetBits(&dst[id], tmp_p, mEndOfBufP, DC::component::code[dIdx], DC::component::size[dIdx]);
				if (dIdx) {
					if (diff < 0)
						diff--;
					SetBits(&dst[id], tmp_p, mEndOfBufP, diff, dIdx);
				}
				run = 0;

				//AC
				for (i = 1; i < 64; i++) {
					absC = abs(src_qua[mid + i]);
					if (absC) {
						while (run > 15) {
							SetBits(&dst[id], tmp_p, mEndOfBufP, AC::component::code[AC::component::ZRL],
								AC::component::size[AC::component::ZRL]);
							run -= 16;
						}
						s = 0;
						while (absC > 0) {
							absC >>= 1;
							s++;
						}
						aIdx = run * 10 + s + (run == 15);
						SetBits(&dst[id], tmp_p, mEndOfBufP, AC::component::code[aIdx],
							AC::component::size[aIdx]);
						v = src_qua[mid + i];
						if (v < 0)
							v--;
						SetBits(&dst[id], tmp_p, mEndOfBufP, v, s);
						run = 0;
					} else {
						if (i == 63) {
							SetBits(&dst[id], tmp_p, mEndOfBufP, AC::component::code[AC::component::EOB],
								AC::component::size[AC::component::EOB]);
						} else
							run++;
					}
				}
			} else {
				//DC
				if (mid == Cbsize) {
					diff = src_qua[mid];
				} else
					diff = src_qua[mid] - src_qua[mid - 64];
				absC = abs(diff);
				dIdx = 0;
				while (absC > 0) {
					absC >>= 1;
					dIdx++;
				}
				SetBits(&dst[id], tmp_p, mEndOfBufP, DC::component::code[dIdx], DC::component::size[dIdx]);
				if (dIdx) {
					if (diff < 0)
						diff--;
					SetBits(&dst[id], tmp_p, mEndOfBufP, diff, dIdx);
				}
				run = 0;

				//AC
				for (i = 1; i < 64; i++) {
					absC = abs(src_qua[mid + i]);
					if (absC) {
						while (run > 15) {
							SetBits(&dst[id], tmp_p, mEndOfBufP, AC::component::code[AC::component::ZRL],
								AC::component::size[AC::component::ZRL]);
							run -= 16;
						}
						s = 0;
						while (absC > 0) {
							absC >>= 1;
							s++;
						}
						aIdx = run * 10 + s + (run == 15);
						SetBits(&dst[id], tmp_p, mEndOfBufP, AC::component::code[aIdx],
							AC::component::size[aIdx]);
						v = src_qua[mid + i];
						if (v < 0)
							v--;
						SetBits(&dst[id], tmp_p, mEndOfBufP, v, s);
						run = 0;
					} else {
						if (i == 63) {
							SetBits(&dst[id], tmp_p, mEndOfBufP, AC::component::code[AC::component::EOB],
								AC::component::size[AC::component::EOB]);
						} else
							run++;
					}
				}
			}
		}

		//完全逐次処理、CPUで行った方が圧倒的に速い
		void cpu_huffman_middle(GPUOutBitStreamState *ImOBSP, int sizeX, int sizeY, byte* num_bits) { //,
			int i;
			const int blsize = (sizeX * sizeY + sizeX * sizeY / 2) / 64; //2*(size/2)*(size/2)

			//BitPosは7が上位で0が下位なので注意,更に位置なので注意。7なら要素は0,0なら要素は7
			ImOBSP[0]._num_bits = ImOBSP[0]._byte_pos * 8 + (7 - ImOBSP[0]._bit_pos);

			//出力用、構造体無駄な要素が入っちゃうので
			num_bits[0] = ImOBSP[0]._num_bits;

			ImOBSP[0]._byte_pos = 0;
			ImOBSP[0]._bit_pos = 7;

			for (i = 1; i < blsize; i++) {

				ImOBSP[i]._num_bits = ImOBSP[i]._byte_pos * 8 + (7 - ImOBSP[i]._bit_pos);

				//出力用、構造体無駄な要素が入っちゃうので
				num_bits[i] = ImOBSP[i]._num_bits;

				ImOBSP[i]._bit_pos = ImOBSP[i - 1]._bit_pos;
				ImOBSP[i]._byte_pos = ImOBSP[i - 1]._byte_pos;

				ImOBSP[i]._bit_pos -= ImOBSP[i - 1]._num_bits % 8;
				//繰り上がり
				if (ImOBSP[i]._bit_pos < 0) {
					ImOBSP[i]._byte_pos++;
					ImOBSP[i]._bit_pos += 8;
				}
				ImOBSP[i]._byte_pos += ImOBSP[i - 1]._num_bits / 8;
			}
		}

		//排他処理のため3つに分ける
		//1MCUは最小4bit(EOBのみ)なので1Byteのバッファに最大3MCUが競合する。だから3つに分ける。
		__global__ void gpu_huffman_write_devide0(GPUOutBitStreamState *mOBSP, byte *mBufP, byte *OmBufP,
			int sizeX, int sizeY) { //,
			int id = (blockIdx.x * blockDim.x + threadIdx.x); //マクロブロック番号
			if (id % 3 == 0) {
				WriteBits(&mOBSP[id], OmBufP, &mBufP[id * MBS], id);
			}
		}
		__global__ void gpu_huffman_write_devide1(GPUOutBitStreamState *mOBSP, byte *mBufP, byte *OmBufP,
			int sizeX, int sizeY) { //,
			int id = (blockIdx.x * blockDim.x + threadIdx.x); //マクロブロック番号
			if (id % 3 == 1) {
				WriteBits(&mOBSP[id], OmBufP, &mBufP[id * MBS], id);
			}
		}
		__global__ void gpu_huffman_write_devide2(GPUOutBitStreamState *mOBSP, byte *mBufP, byte *OmBufP,
			int sizeX, int sizeY) { //,
			int id = (blockIdx.x * blockDim.x + threadIdx.x); //マクロブロック番号
			if (id % 3 == 2) {
				WriteBits(&mOBSP[id], OmBufP, &mBufP[id * MBS], id);
			}
		}

		JpegEncoder::JpegEncoder() :
			_width(0),
			_height(0),
			_y_size(0),
			_c_size(0),
			_ycc_size(0),
			_num_bits(0),
			_out_bit_stream(_ycc_size / 64, MBS),
			_trans_table_Y(0),
			_trans_table_C(0),
			_src(0),
			_yuv_buffer(0),
			_quantized(0),
			_dct_coeficient(0),
			_dct_tmp_buffer(0),
			_grid_color(0, 0, 0),
			_block_color(0, 0, 0),
			_grid_dct(0, 0, 0),
			_block_dct(0, 0, 0),
			_grid_quantize_y(0, 0, 0),
			_block_quantize_y(0, 0, 0),
			_grid_quantize_c(0, 0, 0),
			_block_quantize_c(0, 0, 0),
			_grid_mcu(0, 0, 0),
			_block_mcu(0, 0, 0),
			_grid_huffman(0, 0, 0),
			_block_huffman(0, 0, 0) {

		}

		JpegEncoder::JpegEncoder(size_t width, size_t height) :
			_width(width),
			_height(height),
			_y_size(width * height),
			_c_size(_y_size / 4),
			_ycc_size(_y_size + _c_size * 2),
			_num_bits(_ycc_size / 64),
			_out_bit_stream(_ycc_size / 64, BYTES_PER_MCU),
			_trans_table_Y(_y_size),
			_trans_table_C(_y_size),
			_src(_y_size * 3),
			_yuv_buffer(_ycc_size),
			_quantized(_ycc_size),
			_dct_coeficient(_ycc_size),
			_dct_tmp_buffer(_ycc_size),
			_grid_color(_y_size / THREADS, 1, 1),
			_block_color(THREADS, 1, 1),
			_grid_dct((_ycc_size) / 64 / DCT4_TH, 1, 1),
			_block_dct(DCT4_TH, 8, 8),
			_grid_quantize_y(_y_size / QUA0_TH, 1, 1),
			_block_quantize_y(QUA0_TH, 1, 1),
			_grid_quantize_c((2 * _c_size) / QUA1_TH, 1, 1),
			_block_quantize_c(QUA1_TH, 1, 1),
			_grid_mcu(_ycc_size / 64 / HUF0_TH, 1, 1),
			_block_mcu(HUF0_TH, 1, 1),
			_grid_huffman(_ycc_size / 64 / HUF1_TH, 1, 1),
			_block_huffman(HUF1_TH, 1, 1) {

			make_trans_table(_trans_table_Y.host_data(), _trans_table_C.host_data(), width, height);
			_trans_table_C.sync_to_device();
			_trans_table_Y.sync_to_device();
		}

		void JpegEncoder::setImageSize(size_t width, size_t height) {
			_width = width;
			_height = height;
			_y_size = width * height;
			_c_size = _y_size / 4;
			_ycc_size = _y_size + _c_size * 2;

			_num_bits.resize(_ycc_size / 64);
			_out_bit_stream.resize(_ycc_size / 64, BYTES_PER_MCU);

			_trans_table_Y.resize(_y_size);
			_trans_table_C.resize(_y_size);
			_src.resize(_y_size * 3);
			_yuv_buffer.resize(_ycc_size);
			_quantized.resize(_ycc_size);
			_dct_coeficient.resize(_ycc_size);
			_dct_tmp_buffer.resize(_ycc_size);

			_grid_color = dim3(_y_size / THREADS, 1, 1);
			_block_color = dim3(THREADS, 1, 1);
			_grid_dct = dim3((_ycc_size) / 64 / DCT4_TH, 1, 1);
			_block_dct = dim3(DCT4_TH, 8, 8);
			_grid_quantize_y = dim3(_y_size / QUA0_TH, 1, 1);
			_block_quantize_y = dim3(QUA0_TH, 1, 1);
			_grid_quantize_c = dim3((2 * _c_size) / QUA1_TH, 1, 1);
			_block_quantize_c = dim3(QUA1_TH, 1, 1);
			_grid_mcu = dim3(_ycc_size / 64 / HUF0_TH, 1, 1);
			_block_mcu = dim3(HUF0_TH, 1, 1);
			_grid_huffman = dim3(_ycc_size / 64 / HUF1_TH, 1, 1);
			_block_huffman = dim3(HUF1_TH, 1, 1);
		}

		size_t JpegEncoder::encode(const byte *rgb_data, util::cuda::device_memory<byte> &result) {

			inner_encode(rgb_data);

			// 逐次処理のためCPUに戻す
			_out_bit_stream.status().sync_to_host();
			cpu_huffman_middle(_out_bit_stream.status().host_data(), _width, _height, _num_bits.data());
			_out_bit_stream.status().sync_to_device();

			gpu_huffman_write_devide0<<<_grid_huffman, _block_huffman>>>(
				_out_bit_stream.status().device_data(),
				_out_bit_stream.writable_head(), result.device_data(), _width, _height);
			gpu_huffman_write_devide1<<<_grid_huffman, _block_huffman>>>(
				_out_bit_stream.status().device_data(),
				_out_bit_stream.writable_head(), result.device_data(), _width, _height);
			gpu_huffman_write_devide2<<<_grid_huffman, _block_huffman>>>(
				_out_bit_stream.status().device_data(),
				_out_bit_stream.writable_head(), result.device_data(), _width, _height);

			return _out_bit_stream.available_size();
		}
		size_t JpegEncoder::encode(const byte *rgb_data, JpegOutBitStream &out_bit_stream, ByteBuffer &num_bits) {
			inner_encode(rgb_data);

			// 逐次処理のためCPUに戻す
			out_bit_stream.status().sync_to_host();
			cpu_huffman_middle(out_bit_stream.status().host_data(), _width, _height, num_bits.data());
			out_bit_stream.status().sync_to_device();

			return out_bit_stream.available_size();
		}

		void JpegEncoder::inner_encode(const byte* rgb_data) {
			_src.write_device(rgb_data, _width * _height * 3);

			gpu_color_trans_Y<<<_grid_color, _block_color>>>(
				_src.device_data(), _yuv_buffer.device_data(), _trans_table_Y.device_data());
			gpu_color_trans_C<<<_grid_color, _block_color>>>(
				_src.device_data(), _yuv_buffer.device_data(), _trans_table_C.device_data(),
				_height, _c_size);

			gpu_dct_0<<<_grid_dct, _block_dct>>>(
				_yuv_buffer.device_data(), _dct_tmp_buffer.device_data());
			gpu_dct_1<<<_grid_dct, _block_dct>>>(
				_dct_tmp_buffer.device_data(), _dct_coeficient.device_data());

			gpu_zig_quantize_Y<<<_grid_quantize_y, _block_quantize_y>>>(
				_dct_coeficient.device_data(), _quantized.device_data());
			gpu_zig_quantize_C<<<_grid_quantize_c, _block_quantize_c>>>(
				_dct_coeficient.device_data(), _quantized.device_data(), _y_size);

			gpu_huffman_mcu<<<_grid_mcu, _block_mcu>>>(
				_quantized.device_data(), _out_bit_stream.status().device_data(),
				_out_bit_stream.writable_head(), _out_bit_stream.end(), _width, _height);

		}

		/**
		 * コンストラクタ
		 *
		 * 利用可能な状態にするためには幅と高さをセットする必要がある
		 */
		JpegDecoder::JpegDecoder() :
			_width(0),
			_height(0),
			_y_size(0),
			_c_size(0),
			_ycc_size(0),
			_itrans_table_Y(0),
			_itrans_table_C(0),
			_yuv_buffer(0),
			_quantized(0),
			_dct_coeficient(0),
			_dct_tmp_buffer(0),
			_grid_color(0, 0, 0),
			_block_color(0, 0, 0),
			_grid_dct(0, 0, 0),
			_block_dct(0, 0, 0),
			_grid_quantize_y(0, 0, 0),
			_block_quantize_y(0, 0, 0),
			_grid_quantize_c(0, 0, 0),
			_block_quantize_c(0, 0, 0) {

		}
		/**
		 * コンストラクタ
		 * @param width 幅
		 * @param height 高さ
		 */
		JpegDecoder::JpegDecoder(size_t width, size_t height) :
			_width(width),
			_height(height),
			_y_size(width * height),
			_c_size(_y_size / 4),
			_ycc_size(_y_size + _c_size * 2),
			_itrans_table_Y(_y_size),
			_itrans_table_C(_y_size),
			_yuv_buffer(_ycc_size),
			_quantized(_ycc_size),
			_dct_coeficient(_ycc_size),
			_dct_tmp_buffer(_ycc_size),
			_grid_color(_y_size / THREADS, 1, 1),
			_block_color(THREADS, 1, 1),
			_grid_dct((_ycc_size) / 64 / DCT4_TH, 1, 1),
			_block_dct(DCT4_TH, 8, 8),
			_grid_quantize_y(_y_size / QUA0_TH, 1, 1),
			_block_quantize_y(QUA0_TH, 1, 1),
			_grid_quantize_c((2 * _c_size) / QUA1_TH, 1, 1),
			_block_quantize_c(QUA1_TH, 1, 1) {

			make_itrans_table(_itrans_table_Y.host_data(), _itrans_table_C.host_data(), width, height);
			_itrans_table_C.sync_to_device();
			_itrans_table_Y.sync_to_device();
		}
		/**
		 * デコードするイメージのサイズを指定する
		 * @param width 幅
		 * @param height 高さ
		 */
		void JpegDecoder::setImageSize(size_t width, size_t height) {
			_width = width;
			_height = height;
			_y_size = width * height;
			_c_size = _y_size / 4;
			_ycc_size = _y_size + _c_size * 2;

			_itrans_table_Y.resize(_y_size);
			_itrans_table_C.resize(_y_size);
			_yuv_buffer.resize(_ycc_size);
			_quantized.resize(_ycc_size);
			_dct_coeficient.resize(_ycc_size);
			_dct_tmp_buffer.resize(_ycc_size);

			_grid_color = dim3(_y_size / THREADS, 1, 1);
			_block_color = dim3(THREADS, 1, 1);
			_grid_dct = dim3((_ycc_size) / 64 / DCT4_TH, 1, 1);
			_block_dct = dim3(DCT4_TH, 8, 8);
			_grid_quantize_y = dim3(_y_size / QUA0_TH, 1, 1);
			_block_quantize_y = dim3(QUA0_TH, 1, 1);
			_grid_quantize_c = dim3((2 * _c_size) / QUA1_TH, 1, 1);
			_block_quantize_c = dim3(QUA1_TH, 1, 1);
		}

		/**
		 * デコードする
		 * @param src JpegEncoderにより生成されたソースデータ
		 * @param src_size ソースサイズ
		 * @param result 結果を格納するバッファ
		 * @param result_size 結果バッファの有効なバイト数
		 */
		void JpegDecoder::decode(const byte *src, size_t src_size, util::cuda::device_memory<byte> &result) {
			util::InBitStream mIBSP(src, src_size);

			decode_huffman(&mIBSP, _quantized.host_data(), _width, _height);
			_quantized.sync_to_device();

			gpu_izig_quantize_Y<<<_grid_quantize_y, _block_quantize_y>>>(
				_quantized.device_data(), _dct_coeficient.device_data());
			gpu_izig_quantize_C<<<_grid_quantize_c, _block_quantize_c>>>(
				_quantized.device_data(), _dct_coeficient.device_data(), _y_size);

			gpu_idct_0<<<_grid_dct, _block_dct>>>(
				_dct_coeficient.device_data(), _dct_tmp_buffer.device_data());
			gpu_idct_1<<<_grid_dct, _block_dct>>>(
				_dct_tmp_buffer.device_data(), _yuv_buffer.device_data());

			gpu_color_itrans<<<_grid_color, _block_color>>>(
				_yuv_buffer.device_data(), result.device_data(),
				_itrans_table_Y.device_data(), _itrans_table_C.device_data(), _c_size);
		}

	} // namespace cuda
} // namespace jpeg

