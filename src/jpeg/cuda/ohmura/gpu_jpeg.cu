#include <cstdio>
#include <cstdlib>

#include "gpu_jpeg.cuh"
#include "gpu_out_bit_stream.cuh"
#include "../encoder_tables_device.cuh"

#include "../../cpu/cpu_jpeg.h"

#include "../../../utils/out_bit_stream.h"
#include "../../../utils/in_bit_stream.h"

namespace jpeg {
	namespace ohmura {

		using namespace util;
		using namespace cuda::encode_table;

		__device__ __constant__ static float kDisSqrt2 = 1.0 / 1.41421356; //! 2の平方根の逆数

		__device__ __constant__ static float CosT[] = {
			0.707107, 0.707107, 0.707107, 0.707107, 0.707107, 0.707107, 0.707107, 0.707107, 0.980785, 0.83147,
			0.55557, 0.19509, -0.19509, -0.55557, -0.83147, -0.980785, 0.92388, 0.382683, -0.382683, -0.92388,
			-0.92388, -0.382683, 0.382683, 0.92388, 0.83147, -0.19509, -0.980785, -0.55557, 0.55557, 0.980785,
			0.19509, -0.83147, 0.707107, -0.707107, -0.707107, 0.707107, 0.707107, -0.707107, -0.707107,
			0.707107, 0.55557, -0.980785, 0.19509, 0.83147, -0.83147, -0.19509, 0.980785, -0.55557, 0.382683,
			-0.92388, 0.92388, -0.382683, -0.382683, 0.92388, -0.92388, 0.382683, 0.19509, -0.55557, 0.83147,
			-0.980785, 0.980785, -0.83147, 0.55557, -0.19509 };

		__device__ __constant__ static float ICosT[] = {
			1, 1, 1, 1, 1, 1, 1, 1, 0.980785, 0.83147, 0.55557, 0.19509, -0.19509, -0.55557, -0.83147,
			-0.980785, 0.92388, 0.382683, -0.382683, -0.92388, -0.92388, -0.382683, 0.382683, 0.92388,
			0.83147, -0.19509, -0.980785, -0.55557, 0.55557, 0.980785, 0.19509, -0.83147, 0.707107, -0.707107,
			-0.707107, 0.707107, 0.707107, -0.707107, -0.707107, 0.707107, 0.55557, -0.980785, 0.19509,
			0.83147, -0.83147, -0.19509, 0.980785, -0.55557, 0.382683, -0.92388, 0.92388, -0.382683,
			-0.382683, 0.92388, -0.92388, 0.382683, 0.19509, -0.55557, 0.83147, -0.980785, 0.980785, -0.83147,
			0.55557, -0.19509 };

		__device__ byte revise_value_d(double v) {
			if (v < 0.0)
				return 0;
			if (v > 255.0)
				return 255;
			return (byte) v;
		}

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
				0.1440 * src_img[3 * id] + 0.5870 * src_img[3 * id + 1] + 0.2990 * src_img[3 * id + 2] - 128); //-128
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
					-0.0813 * src_img[3 * id] - 0.4187 * src_img[3 * id + 1] + 0.5000 * src_img[3 * id + 2]); //-128
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
			int id = 64 * (blockIdx.x * blockDim.x + threadIdx.x);
			int y = threadIdx.y, u = threadIdx.z;
			pro_f[id + y * 8 + u] = src_ycc[id + y * 8 + 0] * CosT[u * 8 + 0]
				+ src_ycc[id + y * 8 + 1] * CosT[u * 8 + 1] + src_ycc[id + y * 8 + 2] * CosT[u * 8 + 2]
				+ src_ycc[id + y * 8 + 3] * CosT[u * 8 + 3] + src_ycc[id + y * 8 + 4] * CosT[u * 8 + 4]
				+ src_ycc[id + y * 8 + 5] * CosT[u * 8 + 5] + src_ycc[id + y * 8 + 6] * CosT[u * 8 + 6]
				+ src_ycc[id + y * 8 + 7] * CosT[u * 8 + 7];
		}
		__global__ void gpu_dct_1(float *pro_f, int *dst_coef) {
			int id = 64 * (blockIdx.x * blockDim.x + threadIdx.x);
			int v = threadIdx.y, u = threadIdx.z;
			dst_coef[id + v * 8 + u] = int(
				(pro_f[id + 0 * 8 + u] * CosT[v * 8 + 0] + pro_f[id + 1 * 8 + u] * CosT[v * 8 + 1]
					+ pro_f[id + 2 * 8 + u] * CosT[v * 8 + 2] + pro_f[id + 3 * 8 + u] * CosT[v * 8 + 3]
					+ pro_f[id + 4 * 8 + u] * CosT[v * 8 + 4] + pro_f[id + 5 * 8 + u] * CosT[v * 8 + 5]
					+ pro_f[id + 6 * 8 + u] * CosT[v * 8 + 6] + pro_f[id + 7 * 8 + u] * CosT[v * 8 + 7]) / 4);
		}

		//各ドットについて一気にDCTを行う。全てグローバルメモリ
		__global__ void gpu_idct_0(int *src_ycc, float *pro_f) {
			int id = 64 * (blockIdx.x * blockDim.x + threadIdx.x);
			int v = threadIdx.y, x = threadIdx.z;
			//uが0~7
			pro_f[id + v * 8 + x] = kDisSqrt2 * src_ycc[id + v * 8 + 0] * ICosT[0 * 8 + x] //kDisSqrt2 = Cu
			+ src_ycc[id + v * 8 + 1] * ICosT[1 * 8 + x] + src_ycc[id + v * 8 + 2] * ICosT[2 * 8 + x]
				+ src_ycc[id + v * 8 + 3] * ICosT[3 * 8 + x] + src_ycc[id + v * 8 + 4] * ICosT[4 * 8 + x]
				+ src_ycc[id + v * 8 + 5] * ICosT[5 * 8 + x] + src_ycc[id + v * 8 + 6] * ICosT[6 * 8 + x]
				+ src_ycc[id + v * 8 + 7] * ICosT[7 * 8 + x];
		}
		__global__ void gpu_idct_1(float *pro_f, int *dst_coef) {
			int id = 64 * (blockIdx.x * blockDim.x + threadIdx.x);
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
			dst_qua[64 * (id / 64) + Zigzag::sequence[id % 64]] = src_coef[id] / Quantize::luminance[id % 64];
		}

		__global__ void gpu_zig_quantize_C(int *src_coef, int *dst_qua, int size) {
			int id = blockIdx.x * blockDim.x + threadIdx.x + size;
			dst_qua[64 * (id / 64) + Zigzag::sequence[id % 64]] = src_coef[id] / Quantize::component[id % 64];
		}

		__global__ void gpu_izig_quantize_Y(int *src_qua, int *dst_coef) {
			int id = blockIdx.x * blockDim.x + threadIdx.x;
			dst_coef[id] = src_qua[64 * (id / 64) + Zigzag::sequence[id % 64]] * Quantize::luminance[id % 64];
		}

		__global__ void gpu_izig_quantize_C(int *src_qua, int *dst_coef, int size) {

			int id = blockIdx.x * blockDim.x + threadIdx.x + size;
			dst_coef[id] = src_qua[64 * (id / 64) + Zigzag::sequence[id % 64]] * Quantize::component[id % 64];

		}

		__global__ void gpu_huffman_mcu(int *src_qua, OutBitStreamState *dst, byte *mBufP, byte *mEndOfBufP,
			int sizeX, int sizeY) { //,
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
				SetBits(&dst[id], tmp_p, mEndOfBufP, DC::luminance::code[dIdx],
					DC::luminance::code_size[dIdx]);
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
								AC::luminance::code_size[AC::luminance::ZRL]);
							run -= 16;
						}
						s = 0;
						while (absC > 0) {
							absC >>= 1;
							s++;
						}
						aIdx = run * 10 + s + (run == 15);
						SetBits(&dst[id], tmp_p, mEndOfBufP, AC::luminance::code[aIdx],
							AC::luminance::code_size[aIdx]);
						v = src_qua[mid + i];
						if (v < 0)
							v--;
						SetBits(&dst[id], tmp_p, mEndOfBufP, v, s);
						run = 0;
					} else {
						if (i == 63) {
							SetBits(&dst[id], tmp_p, mEndOfBufP, AC::luminance::code[AC::luminance::EOB],
								AC::luminance::code_size[AC::luminance::EOB]);
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
				SetBits(&dst[id], tmp_p, mEndOfBufP, DC::component::code[dIdx],
					DC::component::code_size[dIdx]);
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
								AC::component::code_size[AC::component::ZRL]);
							run -= 16;
						}
						s = 0;
						while (absC > 0) {
							absC >>= 1;
							s++;
						}
						aIdx = run * 10 + s + (run == 15);
						SetBits(&dst[id], tmp_p, mEndOfBufP, AC::component::code[aIdx],
							AC::component::code_size[aIdx]);
						v = src_qua[mid + i];
						if (v < 0)
							v--;
						SetBits(&dst[id], tmp_p, mEndOfBufP, v, s);
						run = 0;
					} else {
						if (i == 63) {
							SetBits(&dst[id], tmp_p, mEndOfBufP, AC::component::code[AC::component::EOB],
								AC::component::code_size[AC::component::EOB]);
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
				SetBits(&dst[id], tmp_p, mEndOfBufP, DC::component::code[dIdx],
					DC::component::code_size[dIdx]);
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
								AC::component::code_size[AC::component::ZRL]);
							run -= 16;
						}
						s = 0;
						while (absC > 0) {
							absC >>= 1;
							s++;
						}
						aIdx = run * 10 + s + (run == 15);
						SetBits(&dst[id], tmp_p, mEndOfBufP, AC::component::code[aIdx],
							AC::component::code_size[aIdx]);
						v = src_qua[mid + i];
						if (v < 0)
							v--;
						SetBits(&dst[id], tmp_p, mEndOfBufP, v, s);
						run = 0;
					} else {
						if (i == 63) {
							SetBits(&dst[id], tmp_p, mEndOfBufP, AC::component::code[AC::component::EOB],
								AC::component::code_size[AC::component::EOB]);
						} else
							run++;
					}
				}
			}
		}

		//完全逐次処理、CPUで行った方が圧倒的に速い
		void cpu_huffman_middle(OutBitStreamState *state, int sizeX, int sizeY, byte* num_bits) { //,
			int i;
			const int blsize = (sizeX * sizeY + sizeX * sizeY / 2) / 64; //2*(size/2)*(size/2)

			//BitPosは7が上位で0が下位なので注意,更に位置なので注意。7なら要素は0,0なら要素は7
			state[0].num_bits_ = state[0].byte_pos_ * 8 + (7 - state[0].bit_pos_);

			//出力用、構造体無駄な要素が入っちゃうので
			num_bits[0] = state[0].num_bits_;

			state[0].byte_pos_ = 0;
			state[0].bit_pos_ = 7;

			for (i = 1; i < blsize; i++) {

				state[i].num_bits_ = state[i].byte_pos_ * 8 + (7 - state[i].bit_pos_);

				//出力用、構造体無駄な要素が入っちゃうので
				num_bits[i] = state[i].num_bits_;

				state[i].bit_pos_ = state[i - 1].bit_pos_;
				state[i].byte_pos_ = state[i - 1].byte_pos_;

				state[i].bit_pos_ -= state[i - 1].num_bits_ % 8;
				//繰り上がり
				if (state[i].bit_pos_ < 0) {
					state[i].byte_pos_++;
					state[i].bit_pos_ += 8;
				}
				state[i].byte_pos_ += state[i - 1].num_bits_ / 8;
			}
		}

		//排他処理のため3つに分ける
		//1MCUは最小4bit(EOBのみ)なので1Byteのバッファに最大3MCUが競合する。だから3つに分ける。
		__global__ void gpu_huffman_write_devide0(OutBitStreamState *state, const byte *src, byte *dst) { //,
			int id = (blockIdx.x * blockDim.x + threadIdx.x); //マクロブロック番号
			if (id % 3 == 0) {
				WriteBits(&state[id], dst, &src[id * MBS]);
			}
		}
		__global__ void gpu_huffman_write_devide1(OutBitStreamState *state, const byte *src, byte *dst) { //,
			int id = (blockIdx.x * blockDim.x + threadIdx.x); //マクロブロック番号
			if (id % 3 == 1) {
				WriteBits(&state[id], dst, &src[id * MBS]);
			}
		}
		__global__ void gpu_huffman_write_devide2(OutBitStreamState *state, const byte *src, byte *dst) { //,
			int id = (blockIdx.x * blockDim.x + threadIdx.x); //マクロブロック番号
			if (id % 3 == 2) {
				WriteBits(&state[id], dst, &src[id * MBS]);
			}
		}

		JpegEncoder::JpegEncoder() :
			width_(0),
			height_(0),
			y_size_(0),
			c_size_(0),
			ycc_size_(0),
			num_bits_(0),
			out_bit_stream_(ycc_size_ / 64, MBS),
			trans_table_Y_(0),
			trans_table_C_(0),
			src_(0),
			yuv_buffer_(0),
			quantized_(0),
			dct_coeficient_(0),
			dct_tmp_buffer_(0),
			grid_color_(0, 0, 0),
			block_color_(0, 0, 0),
			grid_dct_(0, 0, 0),
			block_dct_(0, 0, 0),
			grid_quantize_y_(0, 0, 0),
			block_quantize_y_(0, 0, 0),
			grid_quantize_c_(0, 0, 0),
			block_quantize_c_(0, 0, 0),
			grid_mcu_(0, 0, 0),
			block_mcu_(0, 0, 0),
			grid_huffman_(0, 0, 0),
			block_huffman_(0, 0, 0) {

		}

		JpegEncoder::JpegEncoder(size_t width, size_t height) :
			width_(width),
			height_(height),
			y_size_(width * height),
			c_size_(y_size_ / 4),
			ycc_size_(y_size_ + c_size_ * 2),
			num_bits_(ycc_size_ / 64),
			out_bit_stream_(ycc_size_ / 64, BYTES_PER_MCU),
			trans_table_Y_(y_size_),
			trans_table_C_(y_size_),
			src_(y_size_ * 3),
			yuv_buffer_(ycc_size_),
			quantized_(ycc_size_),
			dct_coeficient_(ycc_size_),
			dct_tmp_buffer_(ycc_size_),
			grid_color_(y_size_ / THREADS, 1, 1),
			block_color_(THREADS, 1, 1),
			grid_dct_((ycc_size_) / 64 / DCT4_TH, 1, 1),
			block_dct_(DCT4_TH, 8, 8),
			grid_quantize_y_(y_size_ / QUA0_TH, 1, 1),
			block_quantize_y_(QUA0_TH, 1, 1),
			grid_quantize_c_((2 * c_size_) / QUA1_TH, 1, 1),
			block_quantize_c_(QUA1_TH, 1, 1),
			grid_mcu_(ycc_size_ / 64 / HUF0_TH, 1, 1),
			block_mcu_(HUF0_TH, 1, 1),
			grid_huffman_(ycc_size_ / 64 / HUF1_TH, 1, 1),
			block_huffman_(HUF1_TH, 1, 1) {

			make_trans_table(trans_table_Y_.host_data(), trans_table_C_.host_data(), width, height);
			trans_table_C_.sync_to_device();
			trans_table_Y_.sync_to_device();
		}

		void JpegEncoder::setImageSize(size_t width, size_t height) {
			width_ = width;
			height_ = height;
			y_size_ = width * height;
			c_size_ = y_size_ / 4;
			ycc_size_ = y_size_ + c_size_ * 2;

			num_bits_.resize(ycc_size_ / 64);
			out_bit_stream_.resize(ycc_size_ / 64, BYTES_PER_MCU);

			trans_table_Y_.resize(y_size_);
			trans_table_C_.resize(y_size_);
			src_.resize(y_size_ * 3);
			yuv_buffer_.resize(ycc_size_);
			quantized_.resize(ycc_size_);
			dct_coeficient_.resize(ycc_size_);
			dct_tmp_buffer_.resize(ycc_size_);

			grid_color_ = dim3(y_size_ / THREADS, 1, 1);
			block_color_ = dim3(THREADS, 1, 1);
			grid_dct_ = dim3((ycc_size_) / 64 / DCT4_TH, 1, 1);
			block_dct_ = dim3(DCT4_TH, 8, 8);
			grid_quantize_y_ = dim3(y_size_ / QUA0_TH, 1, 1);
			block_quantize_y_ = dim3(QUA0_TH, 1, 1);
			grid_quantize_c_ = dim3((2 * c_size_) / QUA1_TH, 1, 1);
			block_quantize_c_ = dim3(QUA1_TH, 1, 1);
			grid_mcu_ = dim3(ycc_size_ / 64 / HUF0_TH, 1, 1);
			block_mcu_ = dim3(HUF0_TH, 1, 1);
			grid_huffman_ = dim3(ycc_size_ / 64 / HUF1_TH, 1, 1);
			block_huffman_ = dim3(HUF1_TH, 1, 1);
		}

		size_t JpegEncoder::encode(const byte *rgb_data, util::cuda::device_memory<byte> &result) {

			inner_encode(rgb_data);

			gpu_huffman_mcu<<<grid_mcu_, block_mcu_>>>(
				quantized_.device_data(), out_bit_stream_.status().device_data(),
				out_bit_stream_.head_device(), out_bit_stream_.end_device(), width_, height_);

			// 逐次処理のためCPUに戻す
			out_bit_stream_.status().sync_to_host();
			cpu_huffman_middle(out_bit_stream_.status().host_data(), width_, height_, num_bits_.data());
			out_bit_stream_.status().sync_to_device();

			gpu_huffman_write_devide0<<<grid_huffman_, block_huffman_>>>(
				out_bit_stream_.status().device_data(),
				out_bit_stream_.head_device(), result.device_data());
			gpu_huffman_write_devide1<<<grid_huffman_, block_huffman_>>>(
				out_bit_stream_.status().device_data(),
				out_bit_stream_.head_device(), result.device_data());
			gpu_huffman_write_devide2<<<grid_huffman_, block_huffman_>>>(
				out_bit_stream_.status().device_data(),
				out_bit_stream_.head_device(), result.device_data());

			return out_bit_stream_.available_size();
		}

		size_t JpegEncoder::encode(const byte *rgb_data, JpegOutBitStream &out_bit_stream,
			ByteBuffer &num_bits) {
			inner_encode(rgb_data);

			gpu_huffman_mcu<<<grid_mcu_, block_mcu_>>>(
				quantized_.device_data(), out_bit_stream_.status().device_data(),
				out_bit_stream_.head_device(), out_bit_stream_.end_device(), width_, height_);

			return 0;
		}

		void JpegEncoder::inner_encode(const byte* rgb_data) {
			src_.write_device(rgb_data, width_ * height_ * 3);

			gpu_color_trans_Y<<<grid_color_, block_color_>>>(
				src_.device_data(), yuv_buffer_.device_data(), trans_table_Y_.device_data());
			gpu_color_trans_C<<<grid_color_, block_color_>>>(
				src_.device_data(), yuv_buffer_.device_data(), trans_table_C_.device_data(),
				height_, c_size_);

			gpu_dct_0<<<grid_dct_, block_dct_>>>(
				yuv_buffer_.device_data(), dct_tmp_buffer_.device_data());
			gpu_dct_1<<<grid_dct_, block_dct_>>>(
				dct_tmp_buffer_.device_data(), dct_coeficient_.device_data());

			gpu_zig_quantize_Y<<<grid_quantize_y_, block_quantize_y_>>>(
				dct_coeficient_.device_data(), quantized_.device_data());
			gpu_zig_quantize_C<<<grid_quantize_c_, block_quantize_c_>>>(
				dct_coeficient_.device_data(), quantized_.device_data(), y_size_);
		}

		/**
		 * コンストラクタ
		 *
		 * 利用可能な状態にするためには幅と高さをセットする必要がある
		 */
		JpegDecoder::JpegDecoder() :
			width_(0),
			height_(0),
			y_size_(0),
			c_size_(0),
			ycc_size_(0),
			itrans_table_Y_(0),
			itrans_table_C_(0),
			yuv_buffer_(0),
			quantized_(0),
			dct_coeficient_(0),
			dct_tmp_buffer_(0),
			grid_color_(0, 0, 0),
			block_color_(0, 0, 0),
			grid_dct_(0, 0, 0),
			block_dct_(0, 0, 0),
			grid_quantize_y_(0, 0, 0),
			block_quantize_y_(0, 0, 0),
			grid_quantize_c_(0, 0, 0),
			block_quantize_c_(0, 0, 0) {

		}
		/**
		 * コンストラクタ
		 * @param width 幅
		 * @param height 高さ
		 */
		JpegDecoder::JpegDecoder(size_t width, size_t height) :
			width_(width),
			height_(height),
			y_size_(width * height),
			c_size_(y_size_ / 4),
			ycc_size_(y_size_ + c_size_ * 2),
			itrans_table_Y_(y_size_),
			itrans_table_C_(y_size_),
			yuv_buffer_(ycc_size_),
			quantized_(ycc_size_),
			dct_coeficient_(ycc_size_),
			dct_tmp_buffer_(ycc_size_),
			grid_color_(y_size_ / THREADS, 1, 1),
			block_color_(THREADS, 1, 1),
			grid_dct_((ycc_size_) / 64 / DCT4_TH, 1, 1),
			block_dct_(DCT4_TH, 8, 8),
			grid_quantize_y_(y_size_ / QUA0_TH, 1, 1),
			block_quantize_y_(QUA0_TH, 1, 1),
			grid_quantize_c_((2 * c_size_) / QUA1_TH, 1, 1),
			block_quantize_c_(QUA1_TH, 1, 1) {

			make_itrans_table(itrans_table_Y_.host_data(), itrans_table_C_.host_data(), width, height);
			itrans_table_C_.sync_to_device();
			itrans_table_Y_.sync_to_device();
		}
		/**
		 * デコードするイメージのサイズを指定する
		 * @param width 幅
		 * @param height 高さ
		 */
		void JpegDecoder::setImageSize(size_t width, size_t height) {
			width_ = width;
			height_ = height;
			y_size_ = width * height;
			c_size_ = y_size_ / 4;
			ycc_size_ = y_size_ + c_size_ * 2;

			itrans_table_Y_.resize(y_size_);
			itrans_table_C_.resize(y_size_);
			yuv_buffer_.resize(ycc_size_);
			quantized_.resize(ycc_size_);
			dct_coeficient_.resize(ycc_size_);
			dct_tmp_buffer_.resize(ycc_size_);

			grid_color_ = dim3(y_size_ / THREADS, 1, 1);
			block_color_ = dim3(THREADS, 1, 1);
			grid_dct_ = dim3((ycc_size_) / 64 / DCT4_TH, 1, 1);
			block_dct_ = dim3(DCT4_TH, 8, 8);
			grid_quantize_y_ = dim3(y_size_ / QUA0_TH, 1, 1);
			block_quantize_y_ = dim3(QUA0_TH, 1, 1);
			grid_quantize_c_ = dim3((2 * c_size_) / QUA1_TH, 1, 1);
			block_quantize_c_ = dim3(QUA1_TH, 1, 1);

			make_itrans_table(itrans_table_Y_.host_data(), itrans_table_C_.host_data(), width, height);
			itrans_table_C_.sync_to_device();
			itrans_table_Y_.sync_to_device();
		}

		/**
		 * デコードする
		 * @param src JpegEncoderにより生成されたソースデータ
		 * @param src_size ソースサイズ
		 * @param result 結果を格納するバッファ
		 */
		void JpegDecoder::decode(const byte *src, size_t src_size, util::cuda::device_memory<byte> &result) {
			util::InBitStream mIBSP(src, src_size);

			cpu::decode_huffman(&mIBSP, quantized_.host_data(), width_, height_);
			quantized_.sync_to_device();

			gpu_izig_quantize_Y<<<grid_quantize_y_, block_quantize_y_>>>(
				quantized_.device_data(), dct_coeficient_.device_data());
			gpu_izig_quantize_C<<<grid_quantize_c_, block_quantize_c_>>>(
				quantized_.device_data(), dct_coeficient_.device_data(), y_size_);

			gpu_idct_0<<<grid_dct_, block_dct_>>>(
				dct_coeficient_.device_data(), dct_tmp_buffer_.device_data());
			gpu_idct_1<<<grid_dct_, block_dct_>>>(
				dct_tmp_buffer_.device_data(), yuv_buffer_.device_data());

			gpu_color_itrans<<<grid_color_, block_color_>>>(
				yuv_buffer_.device_data(), result.device_data(),
				itrans_table_Y_.device_data(), itrans_table_C_.device_data(), c_size_);
		}

	} // namespace gpu
} // namespace jpeg

