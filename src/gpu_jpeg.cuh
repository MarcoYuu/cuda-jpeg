/*
 * gpu_jpeg.cuh
 *
 *  Created on: 2012/09/22
 *      Author: Yuu Momma
 */

#ifndef GPU_JPEG_H_
#define GPU_JPEG_H_

#include "utils/gpu_out_bit_stream.cuh"
#include "type_definitions.h"

#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __host__
#endif

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
__global__ void gpu_huffman_mcu(int *src_qua, GPUOutBitStream *mOBSP, byte *mBufP, byte *mEndOfBufP,
	int sizeX, int sizeY);

//完全逐次処理、CPUで行った方が圧倒的に速い
void cpu_huffman_middle(GPUOutBitStream *ImOBSP, int sizeX, int sizeY, byte* dst_NumBits);

//排他処理のため3つに分ける
//1MCUは最小4bit(EOBのみ)なので1Byteのバッファに最大3MCUが競合する。だから3つに分ける。
__global__ void gpu_huffman_write_devide0(GPUOutBitStream *mOBSP, byte *mBufP, byte *OmBufP,
	int sizeX, int sizeY);
__global__ void gpu_huffman_write_devide1(GPUOutBitStream *mOBSP, byte *mBufP, byte *OmBufP,
	int sizeX, int sizeY);
__global__ void gpu_huffman_write_devide2(GPUOutBitStream *mOBSP, byte *mBufP, byte *OmBufP,
	int sizeX, int sizeY);

#endif /* GPU_JPEG_H_ */

