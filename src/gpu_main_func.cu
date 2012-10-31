#include <cstdlib>
#include <cstring>
#include <cmath>

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

#include "cpu_jpeg.h"
#include "gpu_jpeg.cuh"

#include "utils/util_cv.h"
#include "utils/timer.h"
#include "utils/cuda_timer.h"
#include "utils/cuda_memory.hpp"
#include "utils/util_cv.h"
#include "utils/encoder_tables.h"
#include "utils/in_bit_stream.h"

using namespace std;

using namespace util;
using namespace util::cuda;
using namespace jpeg;
using namespace jpeg::cuda;

void gpu_exec(const std::string &file_name, const std::string &out_file_name) {
	//----------------------------------------------------------------------------
	// 画像読み込み
	//============================================================================
	BitmapCVUtil source(file_name, BitmapCVUtil::RGB_COLOR);
	const int width = source.getWidth();
	const int height = source.getHeight();
	const int y_size = width * height;
	const int c_size = y_size / 4;
	const int ycc_size = y_size + c_size * 2;

	std::cout << "===============================================" << std::endl;
	std::cout << " Start GPU Encoding & Decoding" << std::endl;
	std::cout << "-----------------------------------------------\n" << std::endl;

	//----------------------------------------------------------------------------
	// カーネルDimension設定
	//============================================================================
	const int THREADS = 256;
	const int DCT4_TH = 1;
	const int QUA0_TH = 64;
	const int QUA1_TH = 64;
	const int HUF0_TH = 16;
	const int HUF1_TH = 4; //divide使うなら最速

	dim3 grid_color(y_size / THREADS, 1, 1), block_color(THREADS, 1, 1);
	dim3 grid_dct((ycc_size) / 64 / DCT4_TH, 1, 1), block_dct(DCT4_TH, 8, 8); //DCT4_THは16が最大
	dim3 grid_quantize_y(y_size / QUA0_TH, 1, 1), block_quantize_y(QUA0_TH, 1, 1);
	dim3 grid_quantize_c((2 * c_size) / QUA1_TH, 1, 1), block_quantize_c(QUA1_TH, 1, 1);
	dim3 grid_mcu(ycc_size / 64 / HUF0_TH, 1, 1), block_mcu(HUF0_TH, 1, 1); //YCC_size
	dim3 grid_huffman(ycc_size / 64 / HUF1_TH, 1, 1), block_huffman(HUF1_TH, 1, 1); //YCC_size

	//----------------------------------------------------------------------------
	// 処理開始
	//============================================================================
	int result_size;
	ByteBuffer num_bits(ycc_size / 64);
	cuda_memory<byte> encode_result(sizeof(byte) * (y_size * 3));
	encode_result.fill_zero();
	{
		jpeg::cuda::JpegEncoder encoder(width, height);

		std::cout << "	-----------------------------------------------" << std::endl;
		std::cout << "	 Encode" << std::endl;
		std::cout << "	-----------------------------------------------" << std::endl;
		float elapsed_time_ms = 0.0f;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		{
			result_size = encoder.encode((byte*) source.getRawData(), encode_result);
			encode_result.sync_to_host();
		}
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed_time_ms, start, stop);
		std::cout << "	" << elapsed_time_ms << "[ms]\n" << std::endl;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	BitmapCVUtil result(width, height, 8, source.getBytePerPixel());
	{
		//----------------------------------------------------------------------------
		// 色変換テーブルの作成 Decode用定数転送
		//============================================================================
		cuda_memory<int> itrans_table_Y(y_size);
		cuda_memory<int> itrans_table_C(y_size);
		make_itrans_table(itrans_table_Y.host_data(), itrans_table_C.host_data(), width, height);
		itrans_table_C.sync_to_device();
		itrans_table_Y.sync_to_device();

		//----------------------------------------------------------------------------
		// メモリ確保
		//============================================================================
		device_memory<byte> src(y_size * 3);
		device_memory<int> yuv_buffer(ycc_size);
		cuda_memory<int> quantized(ycc_size);
		device_memory<int> dct_coeficient(ycc_size);
		device_memory<float> dct_tmp_buffer(ycc_size);

		std::cout << "	-----------------------------------------------" << std::endl;
		std::cout << "	 Decode" << std::endl;
		std::cout << "	-----------------------------------------------" << std::endl;
		float elapsed_time_ms = 0.0f;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		{
			//----------------------------------------------------------------------------
			// メモリ確保
			//============================================================================
			InBitStream mIBSP(encode_result.host_data(), result_size);

			//----------------------------------------------------------------------------
			// ハフマン復号
			//============================================================================
			// CPU
			decode_huffman(&mIBSP, quantized.host_data(), width, height);
			quantized.sync_to_device();

			//----------------------------------------------------------------------------
			// 逆量子化
			//============================================================================
			gpu_izig_quantize_Y<<<grid_quantize_y, block_quantize_y>>>(quantized.device_data(), dct_coeficient.device_data());
			gpu_izig_quantize_C<<<grid_quantize_c, block_quantize_c>>>(quantized.device_data(), dct_coeficient.device_data(), y_size);

			//----------------------------------------------------------------------------
			// 逆DCT
			//============================================================================
			gpu_idct_0<<<grid_dct, block_dct>>>(dct_coeficient.device_data(), dct_tmp_buffer.device_data());
			gpu_idct_1<<<grid_dct, block_dct>>>(dct_tmp_buffer.device_data(), yuv_buffer.device_data());

			//----------------------------------------------------------------------------
			// yuv->RGB
			//============================================================================
			gpu_color_itrans<<<grid_color, block_color>>>(yuv_buffer.device_data(), src.device_data(),
				itrans_table_Y.device_data(), itrans_table_C.device_data(), c_size);

			//----------------------------------------------------------------------------
			// 結果転送
			//============================================================================
			src.copy_host((byte*) result.getRawData(), src.size());
		}
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed_time_ms, start, stop);
		std::cout << "	" << elapsed_time_ms << "[ms]\n" << std::endl;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
	result.saveToFile("gpu_" + out_file_name);

	std::cout << "-----------------------------------------------" << std::endl;
	std::cout << " Finish GPU Encoding & Decoding" << std::endl;
	std::cout << "===============================================\n\n" << std::endl;
}

