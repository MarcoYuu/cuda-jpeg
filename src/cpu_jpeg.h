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

// -------------------------------------------------------------------------
// CPUエンコーダ
// =========================================================================
class JpegEncoder {
public:
	JpegEncoder();
	JpegEncoder(size_t width, size_t height);

	void setImageSize(size_t width, size_t height);

	size_t encode(const byte *rgb_data, size_t src_size, byte *result, size_t result_size);
	size_t encode(const byte *rgb_data, size_t src_size, ByteBuffer &result);
	size_t encode(const ByteBuffer &rgb_data, byte *result, size_t result_size);
	size_t encode(const ByteBuffer &rgb_data, ByteBuffer &result);

private:
	IntBuffer _yuv_data;
	IntBuffer _coefficient;
	IntBuffer _quantized;
	OutBitStream _out_bit;

	size_t _width;
	size_t _height;

private:
	void reset();
	void inner_encode( const byte* rgb_data );
};

// -------------------------------------------------------------------------
// CPUデコーダ
// =========================================================================
class JpegDecoder {
public:
	JpegDecoder();
	JpegDecoder(size_t width, size_t height);

	void setImageSize(size_t width, size_t height);

	void decode(const byte *src, size_t src_size, byte *result, size_t result_size);
	void decode(const byte *src, size_t src_size, ByteBuffer &result);
	void decode(const ByteBuffer &src, byte *result, size_t result_size);
	void decode(const ByteBuffer &src, ByteBuffer &result);

private:
	IntBuffer _yuv_data;
	IntBuffer _coefficient;
	IntBuffer _quantized;

	size_t _width;
	size_t _height;

private:
	void inner_decode( InBitStream *in_bit, byte * result );
};

// -------------------------------------------------------------------------
// 符号化関数
// =========================================================================
void color_trans_rgb_to_yuv(const byte* src_img, int* dst_img, size_t sizeX, size_t sizeY);
void color_trans_yuv_to_rgb(const int *src_img, byte *dst_img, size_t sizeX, size_t sizeY);

void dct(const int *src_ycc, int *dst_coef, size_t sizeX, size_t sizeY);
void idct(const int *src_coef, int *dst_ycc, size_t sizeX, size_t sizeY);

void zig_quantize(const int *src_coef, int *dst_qua, size_t sizeX, size_t sizeY);
void izig_quantize(const int *src_qua, int *dst_coef, size_t sizeX, size_t sizeY);

void encode_huffman(const int *src_qua, OutBitStream *obit_stream, size_t sizeX, size_t sizeY);
void decode_huffman(InBitStream *ibit_stream, int *dst_qua, size_t sizeX, size_t sizeY);

#endif /* CPU_JPEG_H_ */

