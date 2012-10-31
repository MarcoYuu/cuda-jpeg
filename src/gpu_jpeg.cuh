/*
 * gpu_jpeg.cuh
 *
 *  Created on: 2012/09/22
 *      Author: Yuu Momma
 */

#ifndef GPU_JPEG_H_
#define GPU_JPEG_H_

#include "utils/gpu_out_bit_stream.cuh"
#include "type_definitions.h"

namespace jpeg {
	namespace cuda {
		//-------------------------------------------------------------------
		// Jpeg圧縮用の変数まとめクラス
		//===================================================================
		class GPUJpegOutBitStream {
		public:
			typedef util::cuda::cuda_memory<jpeg::cuda::GPUOutBitStreamState> StreamState;
			typedef jpeg::cuda::GPUOutBitStreamBuffer StreamBuffer;

		private:
			StreamBuffer _out_bit_stream_buffer;
			StreamState _out_bit_stream_status;

		public:
			GPUJpegOutBitStream(size_t blocks, size_t block_size) :
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
		__global__ void gpu_huffman_mcu(int *src_qua, jpeg::cuda::GPUOutBitStreamState *mOBSP, byte *mBufP,
			byte *mEndOfBufP, int sizeX, int sizeY);

		//完全逐次処理、CPUで行った方が圧倒的に速い
		void cpu_huffman_middle(jpeg::cuda::GPUOutBitStreamState *ImOBSP, int sizeX, int sizeY, byte* dst_NumBits);

		//排他処理のため3つに分ける
		//1MCUは最小4bit(EOBのみ)なので1Byteのバッファに最大3MCUが競合する。だから3つに分ける。
		__global__ void gpu_huffman_write_devide0(jpeg::cuda::GPUOutBitStreamState *mOBSP, byte *mBufP, byte *OmBufP,
			int sizeX, int sizeY);
		__global__ void gpu_huffman_write_devide1(jpeg::cuda::GPUOutBitStreamState *mOBSP, byte *mBufP, byte *OmBufP,
			int sizeX, int sizeY);
		__global__ void gpu_huffman_write_devide2(jpeg::cuda::GPUOutBitStreamState *mOBSP, byte *mBufP, byte *OmBufP,
			int sizeX, int sizeY);

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
			JpegEncoder() :
				_width(0),
				_height(0),
				_y_size(0),
				_c_size(0),
				_ycc_size(0),
				_num_bits(0),
				_out_bit_stream(_ycc_size / 64, MBS),
				_trans_table_Y(0),
				_trans_table_C(0),
				_src(0),
				_yuv_buffer(0),
				_quantized(0),
				_dct_coeficient(0),
				_dct_tmp_buffer(0),
				_grid_color(0, 0, 0),
				_block_color(0, 0, 0),
				_grid_dct(0, 0, 0),
				_block_dct(0, 0, 0),
				_grid_quantize_y(0, 0, 0),
				_block_quantize_y(0, 0, 0),
				_grid_quantize_c(0, 0, 0),
				_block_quantize_c(0, 0, 0),
				_grid_mcu(0, 0, 0),
				_block_mcu(0, 0, 0),
				_grid_huffman(0, 0, 0),
				_block_huffman(0, 0, 0) {

			}

			/**
			 * コンストラクタ
			 * @param width 幅
			 * @param height 高さ
			 */
			JpegEncoder(size_t width, size_t height) :
				_width(width),
				_height(height),
				_y_size(width * height),
				_c_size(_y_size / 4),
				_ycc_size(_y_size + _c_size * 2),
				_num_bits(_ycc_size / 64),
				_out_bit_stream(_ycc_size / 64, BYTES_PER_MCU),
				_trans_table_Y(_y_size),
				_trans_table_C(_y_size),
				_src(_y_size * 3),
				_yuv_buffer(_ycc_size),
				_quantized(_ycc_size),
				_dct_coeficient(_ycc_size),
				_dct_tmp_buffer(_ycc_size),
				_grid_color(_y_size / THREADS, 1, 1),
				_block_color(THREADS, 1, 1),
				_grid_dct((_ycc_size) / 64 / DCT4_TH, 1, 1),
				_block_dct(DCT4_TH, 8, 8),
				_grid_quantize_y(_y_size / QUA0_TH, 1, 1),
				_block_quantize_y(QUA0_TH, 1, 1),
				_grid_quantize_c((2 * _c_size) / QUA1_TH, 1, 1),
				_block_quantize_c(QUA1_TH, 1, 1),
				_grid_mcu(_ycc_size / 64 / HUF0_TH, 1, 1),
				_block_mcu(HUF0_TH, 1, 1),
				_grid_huffman(_ycc_size / 64 / HUF1_TH, 1, 1),
				_block_huffman(HUF1_TH, 1, 1) {

				make_trans_table(_trans_table_Y.host_data(), _trans_table_C.host_data(), width, height);
				_trans_table_C.sync_to_device();
				_trans_table_Y.sync_to_device();
			}

			/**
			 * エンコードするイメージのサイズを指定する
			 * @param width 幅
			 * @param height 高さ
			 */
			void setImageSize(size_t width, size_t height) {
				_width = width;
				_height = height;
				_y_size = width * height;
				_c_size = _y_size / 4;
				_ycc_size = _y_size + _c_size * 2;

				_num_bits.resize(_ycc_size / 64);
				_out_bit_stream.resize(_ycc_size / 64, BYTES_PER_MCU);

				_trans_table_Y.resize(_y_size);
				_trans_table_C.resize(_y_size);
				_src.resize(_y_size * 3);
				_yuv_buffer.resize(_ycc_size);
				_quantized.resize(_ycc_size);
				_dct_coeficient.resize(_ycc_size);
				_dct_tmp_buffer.resize(_ycc_size);

				_grid_color = dim3(_y_size / THREADS, 1, 1);
				_block_color = dim3(THREADS, 1, 1);
				_grid_dct = dim3((_ycc_size) / 64 / DCT4_TH, 1, 1);
				_block_dct = dim3(DCT4_TH, 8, 8);
				_grid_quantize_y = dim3(_y_size / QUA0_TH, 1, 1);
				_block_quantize_y = dim3(QUA0_TH, 1, 1);
				_grid_quantize_c = dim3((2 * _c_size) / QUA1_TH, 1, 1);
				_block_quantize_c = dim3(QUA1_TH, 1, 1);
				_grid_mcu = dim3(_ycc_size / 64 / HUF0_TH, 1, 1);
				_block_mcu = dim3(HUF0_TH, 1, 1);
				_grid_huffman = dim3(_ycc_size / 64 / HUF1_TH, 1, 1);
				_block_huffman = dim3(HUF1_TH, 1, 1);
			}

			/**
			 * エンコードする
			 * @param rgb_data BGRBGR…なデータ
			 * @param result 結果を格納するバッファ
			 * @param num_bits 結果バッファの各bits
			 * @return エンコードされたサイズ
			 */
			size_t encode(const byte *rgb_data, util::cuda::device_memory<byte> &result) {

				inner_encode(rgb_data);

				// 逐次処理のためCPUに戻す
				_out_bit_stream.status().sync_to_host();
				cpu_huffman_middle(_out_bit_stream.status().host_data(), _width, _height, _num_bits.data());
				_out_bit_stream.status().sync_to_device();

				gpu_huffman_write_devide0<<<_grid_huffman, _block_huffman>>>(
					_out_bit_stream.status().device_data(),
					_out_bit_stream.writable_head(), result.device_data(), _width, _height);
				gpu_huffman_write_devide1<<<_grid_huffman, _block_huffman>>>(
					_out_bit_stream.status().device_data(),
					_out_bit_stream.writable_head(), result.device_data(), _width, _height);
				gpu_huffman_write_devide2<<<_grid_huffman, _block_huffman>>>(
					_out_bit_stream.status().device_data(),
					_out_bit_stream.writable_head(), result.device_data(), _width, _height);

				return _out_bit_stream.available_size();
			}

			/**
			 * エンコードする
			 * @param rgb_data BGRBGR…なデータ
			 * @param out_bit_stream ハフマン符号化されたビット列
			 * @param num_bits 各MCUのビット数
			 * @return
			 */
			size_t encode(const byte *rgb_data, GPUJpegOutBitStream &out_bit_stream, ByteBuffer &num_bits) {
				inner_encode(rgb_data);

				// 逐次処理のためCPUに戻す
				out_bit_stream.status().sync_to_host();
				cpu_huffman_middle(out_bit_stream.status().host_data(), _width, _height, num_bits.data());
				out_bit_stream.status().sync_to_device();

				return out_bit_stream.available_size();
			}

		private:
			size_t _width;
			size_t _height;
			size_t _y_size;
			size_t _c_size;
			size_t _ycc_size;

			ByteBuffer _num_bits;
			GPUJpegOutBitStream _out_bit_stream;

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
			void reset();
			void inner_encode(const byte* rgb_data) {
				_src.write_device(rgb_data, _width * _height * 3);

				gpu_color_trans_Y<<<_grid_color, _block_color>>>(
					_src.device_data(), _yuv_buffer.device_data(), _trans_table_Y.device_data());
				gpu_color_trans_C<<<_grid_color, _block_color>>>(
					_src.device_data(), _yuv_buffer.device_data(), _trans_table_C.device_data(),
					_height, _c_size);

				gpu_dct_0<<<_grid_dct, _block_dct>>>(
					_yuv_buffer.device_data(), _dct_tmp_buffer.device_data());
				gpu_dct_1<<<_grid_dct, _block_dct>>>(
					_dct_tmp_buffer.device_data(), _dct_coeficient.device_data());

				gpu_zig_quantize_Y<<<_grid_quantize_y, _block_quantize_y>>>(
					_dct_coeficient.device_data(), _quantized.device_data());
				gpu_zig_quantize_C<<<_grid_quantize_c, _block_quantize_c>>>(
					_dct_coeficient.device_data(), _quantized.device_data(), _y_size);

				gpu_huffman_mcu<<<_grid_mcu, _block_mcu>>>(
					_quantized.device_data(), _out_bit_stream.status().device_data(),
					_out_bit_stream.writable_head(), _out_bit_stream.end(), _width, _height);
			}
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

			void inner_decode(util::InBitStream *in_bit, byte * result);
		};
	} // namespace cuda
} // namespace jpeg

#endif /* GPU_JPEG_H_ */

