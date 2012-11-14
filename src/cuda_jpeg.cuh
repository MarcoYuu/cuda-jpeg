/*
 * cuda_jpeg.h
 *
 *  Created on: 2012/11/07
 *      Author: momma
 */

#ifndef CUDA_JPEG_H_
#define CUDA_JPEG_H_

#include "type_definitions.h"
#include "utils/cuda_memory.hpp"

namespace jpeg {
	namespace cuda {

		using namespace util;

#ifdef DEBUG
		__global__ void ConvertRGBToYUV(const byte* rgb, byte* yuv_result, size_t width, size_t height,
			size_t block_width, size_t block_height, int *result);

		__global__ void ConvertYUVToRGB(const byte* yuv, byte* rgb_result, size_t width, size_t height,
			size_t block_width, size_t block_height);
#else
		/**
		 * RGBをYUVに変換
		 *
		 * 各ブロックごとに独立したバッファに代入
		 *
		 * - grid(block_width/16, block_height/16, width/block_width * height/block_height)
		 * - block(16, 16, 1)
		 *
		 * @param rgb BGRで保存されたソースデータ
		 * @param yuv_result yuvに変換された結果
		 * @param width もと画像の幅
		 * @param height 元画像の高さ
		 * @param block_width ブロックの幅
		 * @param block_heght ブロックの高さ
		 *
		 */__global__ void ConvertRGBToYUV(const byte* rgb, byte* yuv_result, size_t width, size_t height,
			size_t block_width, size_t block_height);

		/**
		 * YUVをRGBに変換
		 *
		 * 各ブロックごとに独立したバッファに代入
		 *
		 * - grid(block_width/16, block_height/16, width/block_width * height/block_height)
		 * - block(16, 16, 1)
		 *
		 * @param yuv
		 * @param rgb_result
		 * @param width
		 * @param height
		 * @param block_width
		 * @param block_height
		 */__global__ void ConvertYUVToRGB(const byte* yuv, byte* rgb_result, size_t width, size_t height,
			size_t block_width, size_t block_height);
#endif
	}
}

#endif /* CUDA_JPEG_H_ */
