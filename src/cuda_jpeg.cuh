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
		using namespace util::cuda;

		/**
		 * 色変換テーブルを作成する
		 *
		 * @param width もと画像の幅
		 * @param height 元画像の高さ
		 * @param block_width ブロックの幅
		 * @param block_heght ブロックの高さ
		 * @param table テーブル出力
		 */
		void CreateConvertTable(size_t width, size_t height, size_t block_width,
			size_t block_height, device_memory<int> &table);

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
		 */
		void ConvertRGBToYUV(const device_memory<byte> &rgb, device_memory<byte> &yuv_result,
			size_t width, size_t height, size_t block_width, size_t block_height,
			device_memory<int> &table);

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
		 */
		void ConvertYUVToRGB(const device_memory<byte> &yuv, device_memory<byte> &rgb_result,
			size_t width, size_t height, size_t block_width, size_t block_height,
			device_memory<int> &table);
	}
}

#endif /* CUDA_JPEG_H_ */
