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

		/**
		 * @brief Jpeg圧縮用の変数まとめクラス
		 *
		 * @author yuumomma
		 * @version 1.0
		 */
		class JpegOutBitStream {
		public:
			typedef util::cuda::cuda_memory<jpeg::ohmura::OutBitStreamState> StreamState;
			typedef jpeg::ohmura::OutBitStreamBuffer StreamBuffer;

		private:
			StreamBuffer out_bit_stream_buffer_;
			StreamState out_bit_stream_status_;

		public:
			JpegOutBitStream(size_t blocks, size_t block_size) :
				out_bit_stream_buffer_(blocks * block_size),
				out_bit_stream_status_(blocks) {
				out_bit_stream_status_.sync_to_device();
			}

			void resize(size_t blocks, size_t block_size, bool force = false) {
				out_bit_stream_buffer_.resize(blocks * block_size, force);
				out_bit_stream_status_.resize(blocks, force);
				out_bit_stream_status_.sync_to_device();
			}

			StreamState& status() {
				return out_bit_stream_status_;
			}

			const StreamState& status() const {
				return out_bit_stream_status_;
			}

			util::cuda::cuda_memory<byte>& get_stream_buffer() {
				return out_bit_stream_buffer_.get_stream_buffer();
			}

			const util::cuda::cuda_memory<byte>& get_stream_buffer() const {
				return out_bit_stream_buffer_.get_stream_buffer();
			}

			byte* head_device() {
				return out_bit_stream_buffer_.head_device();
			}
			const byte* head_device() const {
				return out_bit_stream_buffer_.head_device();
			}

			byte* end_device() {
				return out_bit_stream_buffer_.end_device();
			}
			const byte* end_device() const {
				return out_bit_stream_buffer_.end_device();
			}

			size_t blocks() const {
				return out_bit_stream_status_.size();
			}

			size_t available_size() const {
				return status()[blocks() - 1].byte_pos_ + (status()[blocks() - 1].bit_pos_ == 7 ? 0 : 1);
			}
		};

		// -------------------------------------------------------------------------
		// GPUエンコーダ
		// =========================================================================
		/**
		 * @brief エンコーダ
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
			 * @return エンコードされたサイズ
			 */
			size_t encode(const byte *rgb_data, util::cuda::device_memory<byte> &result);

			/**
			 * エンコードする
			 * @param rgb_data BGRBGR…なデータ
			 * @param out_bit_stream ハフマン符号化されたビット列
			 * @param num_bits 各MCUのビット数
			 * @return エンコードされたサイズ
			 */
			size_t encode(const byte *rgb_data, JpegOutBitStream &out_bit_stream, ByteBuffer &num_bits);

		private:
			size_t width_;
			size_t height_;
			size_t y_size_;
			size_t c_size_;
			size_t ycc_size_;

			ByteBuffer num_bits_;
			JpegOutBitStream out_bit_stream_;

			util::cuda::cuda_memory<int> trans_table_Y_;
			util::cuda::cuda_memory<int> trans_table_C_;
			util::cuda::device_memory<byte> src_;
			util::cuda::device_memory<int> yuv_buffer_;
			util::cuda::device_memory<int> quantized_;
			util::cuda::device_memory<int> dct_coeficient_;
			util::cuda::device_memory<float> dct_tmp_buffer_;

			dim3 grid_color_, block_color_;
			dim3 grid_dct_, block_dct_;
			dim3 grid_quantize_y_, block_quantize_y_;
			dim3 grid_quantize_c_, block_quantize_c_;
			dim3 grid_mcu_, block_mcu_;
			dim3 grid_huffman_, block_huffman_;

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
		 * @brief Jpegデコーダ
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
			 */
			void decode(const byte *src, size_t src_size, util::cuda::device_memory<byte> &result);

		private:
			size_t width_;
			size_t height_;
			size_t y_size_;
			size_t c_size_;
			size_t ycc_size_;

			util::cuda::cuda_memory<int> itrans_table_Y_;
			util::cuda::cuda_memory<int> itrans_table_C_;
			util::cuda::device_memory<int> yuv_buffer_;
			util::cuda::cuda_memory<int> quantized_;
			util::cuda::device_memory<int> dct_coeficient_;
			util::cuda::device_memory<float> dct_tmp_buffer_;

			dim3 grid_color_, block_color_;
			dim3 grid_dct_, block_dct_;
			dim3 grid_quantize_y_, block_quantize_y_;
			dim3 grid_quantize_c_, block_quantize_c_;

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
		__global__ void gpu_huffman_mcu(int *src_qua, jpeg::ohmura::OutBitStreamState *mOBSP, byte *mBufP,
			byte *mEndOfBufP, int sizeX, int sizeY);

		//完全逐次処理、CPUで行った方が圧倒的に速い
		void cpu_huffman_middle(jpeg::ohmura::OutBitStreamState *ImOBSP, int sizeX, int sizeY,
			byte* dst_NumBits);

		//排他処理のため3つに分ける
		//1MCUは最小4bit(EOBのみ)なので1Byteのバッファに最大3MCUが競合する。だから3つに分ける。
		__global__ void gpu_huffman_write_devide0(jpeg::ohmura::OutBitStreamState *mOBSP, byte *mBufP,
			byte *OmBufP, int sizeX, int sizeY);
		__global__ void gpu_huffman_write_devide1(jpeg::ohmura::OutBitStreamState *mOBSP, byte *mBufP,
			byte *OmBufP, int sizeX, int sizeY);
		__global__ void gpu_huffman_write_devide2(jpeg::ohmura::OutBitStreamState *mOBSP, byte *mBufP,
			byte *OmBufP, int sizeX, int sizeY);
	} // namespace gpu
} // namespace jpeg

#endif /* GPU_JPEG_H_ */

