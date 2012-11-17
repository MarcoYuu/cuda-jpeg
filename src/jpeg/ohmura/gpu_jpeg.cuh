/*
 * gpu_jpeg.cuh
 *
 *  Created on: 2012/09/22
 *      Author: Yuu Momma
 */

#ifndef GPU_JPEG_H_
#define GPU_JPEG_H_

#include "gpu_out_bit_stream.cuh"
#include "../../utils/in_bit_stream.h"
#include "../../utils/type_definitions.h"

namespace jpeg {
	namespace ohmura {

		using namespace util;

		//-------------------------------------------------------------------
		// Jpeg圧縮用の変数まとめクラス
		//===================================================================
		class JpegOutBitStream {
		public:
			typedef util::cuda::cuda_memory<jpeg::ohmura::GPUOutBitStreamState> StreamState;
			typedef jpeg::ohmura::GPUOutBitStreamBuffer StreamBuffer;

		private:
			StreamBuffer _out_bit_stream_buffer;
			StreamState _out_bit_stream_status;

		public:
			JpegOutBitStream(size_t blocks, size_t block_size) :
				_out_bit_stream_buffer(blocks * block_size),
				_out_bit_stream_status(blocks) {
				_out_bit_stream_status.sync_to_device();
			}

			void resize(size_t blocks, size_t block_size, bool force = false) {
				_out_bit_stream_buffer.resize(blocks * block_size, force);
				_out_bit_stream_status.resize(blocks, force);
				_out_bit_stream_status.sync_to_device();
			}

			StreamState& status() {
				return _out_bit_stream_status;
			}

			const StreamState& status() const {
				return _out_bit_stream_status;
			}

			byte* head() {
				return _out_bit_stream_buffer.head();
			}

			byte* end() {
				return _out_bit_stream_buffer.end();
			}

			byte* writable_head() {
				return _out_bit_stream_buffer.writable_head();
			}

			const byte* head() const {
				return _out_bit_stream_buffer.head();
			}

			const byte* end() const {
				return _out_bit_stream_buffer.end();
			}

			const byte* writable_head() const {
				return _out_bit_stream_buffer.writable_head();
			}

			size_t blocks() {
				return _out_bit_stream_status.size();
			}

			size_t available_size() {
				return status()[blocks() - 1]._byte_pos
					+ (status()[blocks() - 1]._bit_pos == 7 ? 0 : 1);
			}
		};

		// -------------------------------------------------------------------------
		// GPUエンコーダ
		// =========================================================================
		/**
		 * エンコーダ
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
			 * @param result 結果を格納するバッファ
			 * @param num_bits 結果バッファの各bits
			 * @return エンコードされたサイズ
			 */
			size_t encode(const byte *rgb_data, util::cuda::device_memory<byte> &result);

			/**
			 * エンコードする
			 * @param rgb_data BGRBGR…なデータ
			 * @param out_bit_stream ハフマン符号化されたビット列
			 * @param num_bits 各MCUのビット数
			 * @return
			 */
			size_t encode(const byte *rgb_data, JpegOutBitStream &out_bit_stream,
				ByteBuffer &num_bits);

		private:
			size_t _width;
			size_t _height;
			size_t _y_size;
			size_t _c_size;
			size_t _ycc_size;

			ByteBuffer _num_bits;
			JpegOutBitStream _out_bit_stream;

			util::cuda::cuda_memory<int> _trans_table_Y;
			util::cuda::cuda_memory<int> _trans_table_C;
			util::cuda::device_memory<byte> _src;
			util::cuda::device_memory<int> _yuv_buffer;
			util::cuda::device_memory<int> _quantized;
			util::cuda::device_memory<int> _dct_coeficient;
			util::cuda::device_memory<float> _dct_tmp_buffer;

			dim3 _grid_color, _block_color;
			dim3 _grid_dct, _block_dct;
			dim3 _grid_quantize_y, _block_quantize_y;
			dim3 _grid_quantize_c, _block_quantize_c;
			dim3 _grid_mcu, _block_mcu;
			dim3 _grid_huffman, _block_huffman;

			static const int THREADS = 256;
			static const int DCT4_TH = 1;
			static const int QUA0_TH = 64;
			static const int QUA1_TH = 64;
			static const int HUF0_TH = 16;
			static const int HUF1_TH = 4;

			static const int BYTES_PER_MCU = 128;

		private:
			void inner_encode(const byte* rgb_data);
		};

		// -------------------------------------------------------------------------
		// GPUデコーダ
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
			 * デコードするイメージのサイズを指定する
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
			void decode(const byte *src, size_t src_size, util::cuda::device_memory<byte> &result);

		private:
			size_t _width;
			size_t _height;
			size_t _y_size;
			size_t _c_size;
			size_t _ycc_size;

			util::cuda::cuda_memory<int> _itrans_table_Y;
			util::cuda::cuda_memory<int> _itrans_table_C;
			util::cuda::device_memory<int> _yuv_buffer;
			util::cuda::cuda_memory<int> _quantized;
			util::cuda::device_memory<int> _dct_coeficient;
			util::cuda::device_memory<float> _dct_tmp_buffer;

			dim3 _grid_color, _block_color;
			dim3 _grid_dct, _block_dct;
			dim3 _grid_quantize_y, _block_quantize_y;
			dim3 _grid_quantize_c, _block_quantize_c;

			static const int THREADS = 256;
			static const int DCT4_TH = 1;
			static const int QUA0_TH = 64;
			static const int QUA1_TH = 64;

			static const int BYTES_PER_MCU = 128;
		};

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
		__global__ void gpu_huffman_mcu(int *src_qua, jpeg::ohmura::GPUOutBitStreamState *mOBSP,
			byte *mBufP, byte *mEndOfBufP, int sizeX, int sizeY);

		//完全逐次処理、CPUで行った方が圧倒的に速い
		void cpu_huffman_middle(jpeg::ohmura::GPUOutBitStreamState *ImOBSP, int sizeX, int sizeY,
			byte* dst_NumBits);

		//排他処理のため3つに分ける
		//1MCUは最小4bit(EOBのみ)なので1Byteのバッファに最大3MCUが競合する。だから3つに分ける。
		__global__ void gpu_huffman_write_devide0(jpeg::ohmura::GPUOutBitStreamState *mOBSP,
			byte *mBufP, byte *OmBufP, int sizeX, int sizeY);
		__global__ void gpu_huffman_write_devide1(jpeg::ohmura::GPUOutBitStreamState *mOBSP,
			byte *mBufP, byte *OmBufP, int sizeX, int sizeY);
		__global__ void gpu_huffman_write_devide2(jpeg::ohmura::GPUOutBitStreamState *mOBSP,
			byte *mBufP, byte *OmBufP, int sizeX, int sizeY);
	} // namespace gpu
} // namespace jpeg

#endif /* GPU_JPEG_H_ */

