/*
 * gpu_jpeg.cuh
 *
 *  Created on: 2012/09/22
 *      Author: Yuu Momma
 */

#ifndef GPU_JPEG_H_
#define GPU_JPEG_H_

#include "utils/gpu_out_bit_stream.cuh"
#include "utils/in_bit_stream.h"
#include "type_definitions.h"

namespace jpeg {
	namespace cuda {
		//-------------------------------------------------------------------
		// Jpeg圧縮用の変数まとめクラス
		//===================================================================
		class JpegOutBitStream {
		public:
			typedef util::cuda::cuda_memory<jpeg::cuda::GPUOutBitStreamState> StreamState;
			typedef jpeg::cuda::GPUOutBitStreamBuffer StreamBuffer;

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
				return status()[blocks() - 1]._byte_pos + (status()[blocks() - 1]._bit_pos == 7 ? 0 : 1);
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
			size_t encode(const byte *rgb_data, JpegOutBitStream &out_bit_stream, ByteBuffer &num_bits);

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
	} // namespace cuda
} // namespace jpeg

#endif /* GPU_JPEG_H_ */

