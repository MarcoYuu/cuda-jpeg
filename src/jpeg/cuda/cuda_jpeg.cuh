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

		/**
		 * @brief 変換テーブルの要素
		 *
		 * @author yuumomma
		 * @version 1.0
		 */
		struct TableElementSrcToDst {
			size_t y; /// 輝度
			size_t u; /// 色差
			size_t v; /// 色差
		};

		typedef device_memory<TableElementSrcToDst> DeviceTable; 	/// デバイスメモリテーブル
		typedef cuda_memory<TableElementSrcToDst> CudaTable; 		/// ホスト同期可能テーブル

		typedef device_memory<byte> DeviceByteBuffer; 				/// デバイスメモリバイトバッファ
		typedef cuda_memory<byte> CudaByteBuffer;					/// ホスト同期可能バイトバッファ

		typedef device_memory<int> DeviceIntBuffer; 				/// デバイスメモリ整数バッファ
		typedef cuda_memory<int> CudaIntBuffer;					/// ホスト同期可能整数バッファ

		typedef device_memory<float> DevicefloatBuffer; 			/// デバイスメモリfloatバッファ
		typedef cuda_memory<float> CudafloatBuffer;				/// ホスト同期可能floatバッファ

		/**
		 * @brief 色変換テーブルを作成する
		 *
		 * pixel番号→Y書き込み位置のマップを作成
		 *
		 * @param width もと画像の幅
		 * @param height 元画像の高さ
		 * @param block_width ブロックの幅
		 * @param block_height ブロックの高さ
		 * @param table テーブル出力
		 */
		void CreateConversionTable(size_t width, size_t height, size_t block_width, size_t block_height,
			DeviceTable &table);

		/**
		 * @brief RGBをYUVに変換
		 *
		 * 各ブロックごとに独立したバッファに代入
		 *
		 * @param rgb BGRで保存されたソースデータ
		 * @param yuv_result yuvに変換された結果
		 * @param width もと画像の幅
		 * @param height 元画像の高さ
		 * @param block_width ブロックの幅
		 * @param block_height ブロックの高さ
		 * @param table 変換テーブル
		 */
		void ConvertRGBToYUV(const DeviceByteBuffer &rgb, DeviceByteBuffer &yuv_result, size_t width,
			size_t height, size_t block_width, size_t block_height, const DeviceTable &table);

		/**
		 * @brief YUVをRGBに変換
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
		void ConvertYUVToRGB(const DeviceByteBuffer &yuv, DeviceByteBuffer &rgb_result, size_t width,
			size_t height, size_t block_width, size_t block_height, const DeviceTable &table);

		/**
		 * @brief DCTを適用する
		 *
		 * @param yuv 64byte=8x8blockごとに連続したメモリに保存されたデータ
		 * @param dct_coefficient DCT係数
		 */
		void DiscreteCosineTransform(const DeviceByteBuffer &yuv, DeviceIntBuffer &dct_coefficient);

		/**
		 * @brief iDCTを適用する
		 *
		 * @param dct_coefficient DCT係数
		 * @param yuv_result 64byte=8x8blockごとに連続したメモリに保存されたデータ
		 */
		void InverseDiscreteCosineTransform(const DeviceIntBuffer &dct_coefficient,
			DeviceByteBuffer &yuv_result);

		/**
		 * @brief DCT用行列の作成
		 *
		 * @param dct_mat DCT計算用行列
		 */
		void CalculateDCTMatrix(float *dct_mat);

		/**
		 * @brief iDCT用行列の作成
		 *
		 * @param idct_mat iDCT計算用行列(転置DCT計算用行列)
		 */
		void CalculateiDCTMatrix(float *idct_mat);

		/**
		 * @brief ジグザグ量子化
		 *
		 * 量子化した上で、ハフマン符号化しやすいように並び替えを行う。
		 *
		 * @param dct_coefficient DCT係数行列
		 * @param quantized 量子化データ
		 * @param block_size ブロックごとの要素数
		 * @param quarity 量子化品質[0,100]
		 */
		void ZigzagQuantize(const DeviceIntBuffer &dct_coefficient, DeviceIntBuffer &quantized,
			int block_size, int quarity = 50);

		/**
		 * @brief 逆ジグザグ量子化
		 *
		 * 逆量子化し、DCTの並びに変える。品質は量子化時と揃えること。
		 *
		 * @param quantized 量子化データ
		 * @param dct_coefficient DCT係数行列
		 * @param block_size ブロックごとの要素数
		 * @param quarity 量子化品質[0,100]
		 */
		void InverseZigzagQuantize(const DeviceIntBuffer &quantized, DeviceIntBuffer &dct_coefficient,
			int block_size, int quarity = 50);
	}
}

#endif /* CUDA_JPEG_H_ */
