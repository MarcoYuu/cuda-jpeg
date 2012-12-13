/*
 * cuda_jpeg.h
 *
 *  Created on: 2012/11/07
 *      Author: momma
 */

#ifndef CUDA_JPEG_H_
#define CUDA_JPEG_H_

#include <utils/type_definitions.h>
#include <utils/cuda/cuda_memory.hpp>

namespace jpeg {
	namespace cuda {

		using namespace util;
		using namespace util::cuda;

		using util::u_int;

		//-------------------------------------------------------------------------------------------------//
		//
		// 方定義
		//
		//-------------------------------------------------------------------------------------------------//
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

		typedef device_memory<TableElementSrcToDst> DeviceTable; /// デバイスメモリテーブル
		typedef cuda_memory<TableElementSrcToDst> CudaTable; /// ホスト同期可能テーブル

		typedef device_memory<byte> DeviceByteBuffer; /// デバイスメモリバイトバッファ
		typedef cuda_memory<byte> CudaByteBuffer; /// ホスト同期可能バイトバッファ

		typedef device_memory<int> DeviceIntBuffer; /// デバイスメモリ整数バッファ
		typedef cuda_memory<int> CudaIntBuffer; /// ホスト同期可能整数バッファ

		typedef device_memory<float> DevicefloatBuffer; /// デバイスメモリfloatバッファ
		typedef cuda_memory<float> CudafloatBuffer; /// ホスト同期可能floatバッファ

		//-------------------------------------------------------------------------------------------------//
		//
		// 符号化クラス
		//
		//-------------------------------------------------------------------------------------------------//
		/**
		 * @brief JpegEncoder
		 *
		 * Jpeg圧縮の補助
		 *
		 * @author yuumomma
		 * @version 1.0
		 */
		class Encoder {
		public:
			/**
			 * @brief コンストラクタ
			 *
			 * @param width 画像幅
			 * @param height 画像高さ
			 */
			Encoder(u_int width, u_int height);

			/**
			 * @brief コンストラクタ
			 *
			 * @param width 画像幅
			 * @param height 画像高さ
			 * @param block_width ブロック幅
			 * @param block_height ブロック高さ
			 */
			Encoder(u_int width, u_int height, u_int block_width, u_int block_height);

			/**
			 * デストラクタ
			 */
			~Encoder();

			/**
			 * @brief 状態のリセット
			 *
			 * 画像、ブロック幅を変更した際はこのメソッドを呼び出すこと
			 */
			void reset();

			/**
			 * @brief 画像サイズを設定する
			 *
			 * @param width 幅
			 * @param height 高さ
			 */
			void setImageSize(u_int width, u_int height);

			/**
			 * @brief ブロック分割幅を指定する
			 *
			 * @param block_width 幅
			 * @param block_height 高さ
			 */
			void setBlockSize(u_int block_width, u_int block_height);

			/**
			 * @brief 圧縮品質の設定
			 *
			 * @param quarity 品質[0-100]
			 */
			void setQuarity(u_int quarity);

			/**
			 *
			 * @return
			 */
			u_int getBlockNum() const;

			/**
			 *
			 * @param rgb
			 * @param huffman
			 * @param effective_bits
			 */
			void encode(const byte* rgb, DeviceByteBuffer &huffman, IntBuffer &effective_bits);

		private:
			class Impl;
			Impl *impl;

			Encoder(const Encoder&);
			Encoder& operator=(const Encoder&);
		};

		/**
		 * @brief JpegDecoder
		 *
		 * Jpeg圧縮の補助
		 *
		 * @author yuumomma
		 * @version 1.0
		 */
		class Decoder {
		public:
			/**
			 * @brief コンストラクタ
			 *
			 * @param width 画像幅
			 * @param height 画像高さ
			 */
			Decoder(u_int width, u_int height);

			/**
			 * デストラクタ
			 */
			~Decoder();

			/**
			 * @brief 状態のリセット
			 *
			 * 画像、ブロック幅を変更した際はこのメソッドを呼び出すこと
			 */
			void reset();

			/**
			 * @brief 画像サイズを設定する
			 *
			 * @param width 幅
			 * @param height 高さ
			 */
			void setImageSize(u_int width, u_int height);

			/**
			 * @brief 圧縮品質の設定
			 *
			 * @param quarity 品質[0-100]
			 */
			void setQuarity(u_int quarity);

			/**
			 * @brief デコードする
			 *
			 * @param rgb 色データ
			 */
			void decode(const byte *huffman, byte *dst);

		private:
			class Impl;
			Impl *impl;

			Decoder(const Decoder&);
			Decoder& operator=(const Decoder&);
		};

		//-------------------------------------------------------------------------------------------------//
		//
		// 個別変換関数
		//
		//-------------------------------------------------------------------------------------------------//
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
			u_int block_size, u_int quarity = 80);

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
			u_int block_size, u_int quarity = 80);

		/**
		 * @brief ハフマンエンコードする
		 *
		 * ハフマン符号化された結果をバッファに書き込む。
		 * ただし、バッファはハフマン符号化されたデータが十分入る大きさでなければならず、
		 * 量子化結果はeffective_bits.size()個のブロックに分割されたデータとみなし書き込みを行う。
		 *
		 * - 第一引数には量子化されたデータを渡す.
		 * - 第二引数には結果を保存するバッファを渡す.
		 * - ただしこのバッファは第三引数の要素数に等分割されて結果が保存される。
		 * - 第三引数は第二引数の各バッファの有効bit数を返す。
		 *
		 * @param quantized 量子化データ
		 * @param result 結果
		 * @param block_size 有効ビット数
		 */
		void HuffmanEncode(const DeviceIntBuffer &quantized, DeviceByteBuffer &result,
			IntBuffer &effective_bits);

		/**
		 * @brief ハフマン符号をデコードする
		 *
		 * @param huffman ハフマン符号
		 * @param quantized 量子化データ
		 * @param width 画像幅
		 * @param height 画像高さ
		 */
		void HuffmanDecode(const ByteBuffer &huffman, IntBuffer &quantized, size_t width, size_t height);
	}
}

#endif /* CUDA_JPEG_H_ */
