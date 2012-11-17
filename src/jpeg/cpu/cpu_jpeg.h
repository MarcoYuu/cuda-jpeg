/*
 * cpu_jpeg.h
 *
 *  Created on: 2012/09/21
 *      Author: Yuu Momma
 */

#ifndef CPU_JPEG_H_
#define CPU_JPEG_H_

#include <vector>

#include "../../utils/out_bit_stream.h"
#include "../../utils/in_bit_stream.h"
#include "../../utils/type_definitions.h"

namespace jpeg {

	using namespace util;

	// -------------------------------------------------------------------------
	// CPUエンコーダ
	// =========================================================================
	/**
	 * CPUエンコーダ
	 *
	 * @author yuumomma
	 * @version 1.0
	 */
	class JpegEncoder {
	public:
		/**
		 * コンストラクタ
		 *
		 * 利用可能な状態にするためには幅と高さをセットする必要がある
		 */
		JpegEncoder();
		/**
		 * コンストラクタ
		 * @param width 幅
		 * @param height 高さ
		 */
		JpegEncoder(size_t width, size_t height);

		/**
		 * エンコードするイメージのサイズを指定する
		 * @param width 幅
		 * @param height 高さ
		 */
		void setImageSize(size_t width, size_t height);

		/**
		 * エンコードする
		 * @param rgb_data BGRBGR…なデータ
		 * @param src_size ソースデータサイズ
		 * @param result 結果を格納するバッファ
		 * @param result_size 結果バッファの有効なバイト数
		 * @return エンコードされたサイズ
		 */
		size_t encode(const byte *rgb_data, size_t src_size, byte *result, size_t result_size);
		/**
		 * エンコードする
		 * @param rgb_data BGRBGR…なデータ
		 * @param src_size ソースデータサイズ
		 * @param result 結果を格納するバッファ
		 * @return エンコードされたサイズ
		 */
		size_t encode(const byte *rgb_data, size_t src_size, ByteBuffer &result);
		/**
		 * エンコードする
		 * @param rgb_data BGRBGR…なデータ
		 * @param result 結果を格納するバッファ
		 * @param result_size　結果バッファの有効なバイト数
		 * @return エンコードされたサイズ
		 */
		size_t encode(const ByteBuffer &rgb_data, byte *result, size_t result_size);
		/**
		 * エンコードする
		 * @param rgb_data BGRBGR…なデータ
		 * @param result 結果を格納するバッファ
		 * @return エンコードされたサイズ
		 */
		size_t encode(const ByteBuffer &rgb_data, ByteBuffer &result);

	private:
		IntBuffer _yuv_data;
		IntBuffer _coefficient;
		IntBuffer _quantized;
		util::OutBitStream _out_bit;

		size_t _width;
		size_t _height;

	private:
		void reset();
		void inner_encode(const byte* rgb_data);
	};

	// -------------------------------------------------------------------------
	// CPUデコーダ
	// =========================================================================
	/**
	 * Jpegデコーダ
	 *
	 * @author yuumomma
	 * @version 1.0
	 */
	class JpegDecoder {
	public:
		/**
		 * コンストラクタ
		 *
		 * 利用可能な状態にするためには幅と高さをセットする必要がある
		 */
		JpegDecoder();
		/**
		 * コンストラクタ
		 * @param width 幅
		 * @param height 高さ
		 */
		JpegDecoder(size_t width, size_t height);

		/**
		 * エンコードするイメージのサイズを指定する
		 * @param width 幅
		 * @param height 高さ
		 */
		void setImageSize(size_t width, size_t height);

		/**
		 * デコードする
		 * @param src JpegEncoderにより生成されたソースデータ
		 * @param src_size ソースサイズ
		 * @param result 結果を格納するバッファ
		 * @param result_size 結果バッファの有効なバイト数
		 */
		void decode(const byte *src, size_t src_size, byte *result, size_t result_size);
		/**
		 * デコードする
		 * @param src JpegEncoderにより生成されたソースデータ
		 * @param src_size ソースサイズ
		 * @param result 結果を格納するバッファ
		 */
		void decode(const byte *src, size_t src_size, ByteBuffer &result);
		/**
		 * デコードする
		 * @param src JpegEncoderにより生成されたソースデータ
		 * @param result 結果を格納するバッファ
		 * @param result_size 結果バッファの有効なバイト数
		 */
		void decode(const ByteBuffer &src, byte *result, size_t result_size);
		/**
		 * デコードする
		 * @param src JpegEncoderにより生成されたソースデータ
		 * @param result 結果を格納するバッファ
		 */
		void decode(const ByteBuffer &src, ByteBuffer &result);

	private:
		IntBuffer _yuv_data;
		IntBuffer _coefficient;
		IntBuffer _quantized;

		size_t _width;
		size_t _height;

	private:
		void inner_decode(util::InBitStream *in_bit, byte * result);
	};

	// -------------------------------------------------------------------------
	// 符号化関数
	// =========================================================================
	/**
	 * RGB→YUV変換
	 * @param src_img ソース
	 * @param dst_img 出力
	 * @param sizeX 幅
	 * @param sizeY 高さ
	 */
	void color_trans_rgb_to_yuv(const byte* src_img, int* dst_img, size_t sizeX, size_t sizeY);
	/**
	 * YUV→RGB変換
	 * @param src_img ソース
	 * @param dst_img 出力
	 * @param sizeX 幅
	 * @param sizeY 高さ
	 */
	void color_trans_yuv_to_rgb(const int *src_img, byte *dst_img, size_t sizeX, size_t sizeY);

	/**
	 * DCT
	 * @param src_ycc YUVデータ
	 * @param dst_coef 出力DCT係数
	 * @param sizeX 幅
	 * @param sizeY 高さ
	 */
	void dct(const int *src_ycc, int *dst_coef, size_t sizeX, size_t sizeY);
	/**
	 * 逆DCT
	 * @param src_coef DCT係数
	 * @param dst_ycc 出力YUV
	 * @param sizeX 幅
	 * @param sizeY 高さ
	 */void idct(const int *src_coef, int *dst_ycc, size_t sizeX, size_t sizeY);

	/**
	 * 量子化
	 * @param src_coef DCT係数
	 * @param dst_qua 量子化結果
	 * @param sizeX 幅
	 * @param sizeY 高さ
	 */
	void zig_quantize(const int *src_coef, int *dst_qua, size_t sizeX, size_t sizeY);
	/**
	 * 逆量子化
	 * @param src_qua 量子化されたデータ
	 * @param dst_coef DCT係数
	 * @param sizeX 幅
	 * @param sizeY 高さ
	 */
	void izig_quantize(const int *src_qua, int *dst_coef, size_t sizeX, size_t sizeY);

	/**
	 * ハフマン符号化
	 * @param src_qua 量子化データ
	 * @param obit_stream 出力ビットストリーム
	 * @param sizeX 幅
	 * @param sizeY 高さ
	 */
	void encode_huffman(const int *src_qua, util::OutBitStream *obit_stream, size_t sizeX,
		size_t sizeY);
	/**
	 * ハフマン復号
	 * @param ibit_stream ハフマンデータストリーム
	 * @param dst_qua 量子化データ
	 * @param sizeX 幅
	 * @param sizeY 高さ
	 */
	void decode_huffman(util::InBitStream *ibit_stream, int *dst_qua, size_t sizeX, size_t sizeY);
} // namespace jpeg

#endif /* CPU_JPEG_H_ */

