/*
 * cuda_jpeg.h
 *
 *  Created on: 2012/11/07
 *      Author: momma
 */

#ifndef CUDA_JPEG_H_
#define CUDA_JPEG_H_

#include "../../utils/type_definitions.h"
#include "../../utils/cuda/cuda_memory.hpp"

namespace jpeg {
	namespace cuda {

		using namespace util;
		using namespace util::cuda;

		struct TableElementSrcToDst {
			size_t y;
			size_t u;
			size_t v;
		};

		typedef device_memory<TableElementSrcToDst> DeviceTable;
		typedef cuda_memory<TableElementSrcToDst> CudaTable;

		typedef device_memory<byte> DeviceByteBuffer;
		typedef cuda_memory<byte> CudaByteBuffer;

		typedef device_memory<int> DeviceIntBuffer;
		typedef cuda_memory<int> CudaIntBuffer;

		typedef device_memory<float> DevicefloatBuffer;
		typedef cuda_memory<float> CudafloatBuffer;

		/**
		 * 色変換テーブルを作成する
		 *
		 * pixel番号→Y書き込み位置のマップを作成
		 *
		 * - grid(block_width/16, block_height/16, width/block_width * height/block_height)
		 * - block(16, 16, 1)
		 *
		 * @param width もと画像の幅
		 * @param height 元画像の高さ
		 * @param block_width ブロックの幅
		 * @param block_height ブロックの高さ
		 * @param table テーブル出力
		 */
		void CreateConversionTable(size_t width, size_t height, size_t block_width,
			size_t block_height, DeviceTable &table);

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
		 * @param block_height ブロックの高さ
		 * @param table 変換テーブル
		 */
		void ConvertRGBToYUV(const DeviceByteBuffer &rgb, DeviceByteBuffer &yuv_result,
			size_t width, size_t height, size_t block_width, size_t block_height,
			const DeviceTable &table);

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
		 * @param block_height ブロックの高さ
		 * @param table 変換テーブル
		 */
		void ConvertYUVToRGB(const DeviceByteBuffer &yuv, DeviceByteBuffer &rgb_result,
			size_t width, size_t height, size_t block_width, size_t block_height,
			const DeviceTable &table);

		void DiscreteCosineTransform(const DeviceByteBuffer &yuv, DeviceIntBuffer &dct_coefficient,
			size_t width, size_t height, size_t block_width, size_t block_height);

		void InverseDiscreteCosineTransform(const DeviceIntBuffer &dct_coefficient, DeviceByteBuffer &yuv_result,
			size_t width, size_t height, size_t block_width, size_t block_height);

		void CalculateDCTMatrix(float *dct_mat);

		void CalculateiDCTMatrix(float *idct_mat);
	}
}

#endif /* CUDA_JPEG_H_ */
