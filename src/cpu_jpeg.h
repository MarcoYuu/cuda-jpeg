/*
 * cpu_jpeg.h
 *
 *  Created on: 2012/09/21
 *      Author: Yuu Momma
 */

#ifndef CPU_JPEG_H_
#define CPU_JPEG_H_

#include <vector>

#include "utils/out_bit_stream.h"
#include "utils/in_bit_stream.h"

#include "type_definitions.h"

void color_trans_rgb_to_yuv(byte* src_img, int* dst_img, int sizeX, int sizeY);
void color_trans_yuv_to_rgb(int *src_img, byte *dst_img, int sizeX, int sizeY);

void dct(int *src_ycc, int *dst_coef, int sizeX, int sizeY);
void idct(int *src_coef, int *dst_ycc, int sizeX, int sizeY);

void zig_quantize(int *src_coef, int *dst_qua, int sizeX, int sizeY);
void izig_quantize(int *src_qua, int *dst_coef, int sizeX, int sizeY);

void huffman_encode(int *src_qua, OutBitStream *obit_stream, int sizeX,
	int sizeY);
void decode_huffman(InBitStream *ibit_stream, int *dst_qua, int sizeX,
	int sizeY);

#endif /* CPU_JPEG_H_ */

