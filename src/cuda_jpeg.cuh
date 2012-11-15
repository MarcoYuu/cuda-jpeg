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
		 * - grid(block_width/16, block_height/16, width/block_width * height/block_height)
		 * - block(16, 16, 1)
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
		 * カーネル起動は各ピクセルごと。
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
		 * @param table 変換テーブル
		 */
		void ConvertRGBToYUV(const device_memory<byte> &rgb, device_memory<byte> &yuv_result,
			size_t width, size_t height, size_t block_width, size_t block_height,
			const device_memory<int> &table);

		/**
		 * YUVをRGBに変換
		 *
		 * 各ブロックごとに独立したバッファに代入
		 *
		 * @param yuv YUV411で保存されたソースデータ
		 * @param rgb_result rgbに変換された結果
		 * @param width もと画像の幅
		 * @param height 元画像の高さ
		 * @param block_width ブロックの幅
		 * @param block_heght ブロックの高さ
		 * @param table 変換テーブル
		 */
		void ConvertYUVToRGB(const device_memory<byte> &yuv, device_memory<byte> &rgb_result,
			size_t width, size_t height, size_t block_width, size_t block_height,
			device_memory<int> &table);
	}
}

#endif /* CUDA_JPEG_H_ */
