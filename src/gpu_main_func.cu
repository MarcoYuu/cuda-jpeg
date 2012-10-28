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
#include "utils/gpu_in_bit_stream.cuh"

using namespace std;

void gpu_exec(const std::string &file_name, const std::string &out_file_name) {
	//----------------------------------------------------------------------------
	// 画像読み込み
	//============================================================================
	BitmapCVUtil source(file_name, BitmapCVUtil::RGB_COLOR);
	const int width = source.getWidth();
	const int height = source.getHeight();
	const int Y_size = width * height;
	const int C_size = Y_size / 4;
	const int YCC_size = Y_size + C_size * 2;

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

	dim3 Dg0_0(Y_size / THREADS, 1, 1), Db0_0(THREADS, 1, 1);
	dim3 Dg0_1(Y_size / THREADS / 2, 1, 1), Db0_1(height / 2, 1, 1);
	dim3 Dg1((YCC_size) / 64 / DCT4_TH, 1, 1), Db1(DCT4_TH, 8, 8); //DCT4_THは16が最大
	dim3 Dg2_0(Y_size / QUA0_TH, 1, 1), Db2_0(QUA0_TH, 1, 1);
	dim3 Dg2_1((2 * C_size) / QUA1_TH, 1, 1), Db2_1(QUA1_TH, 1, 1);
	dim3 Dg3_0(YCC_size / 64 / HUF0_TH, 1, 1), Db3_0(HUF0_TH, 1, 1); //YCC_size
	dim3 Dg3_1(YCC_size / 64 / HUF1_TH, 1, 1), Db3_1(HUF1_TH, 1, 1); //YCC_size

	//----------------------------------------------------------------------------
	// 処理開始
	//============================================================================
	CudaStopWatch watch;

	int result_size;
	ByteBuffer num_bits(YCC_size / 64);
	CudaMemory<byte> encode_result(sizeof(byte) * (Y_size * 3));
	encode_result.fillZero();
	{
		//----------------------------------------------------------------------------
		// 色変換テーブルの作成 Encode用定数転送
		//============================================================================
		CudaMemory<int> trans_table_Y(width * height);
		CudaMemory<int> trans_table_C(width * height);
		make_trans_table(trans_table_Y.host_data(), trans_table_C.host_data(), width, height);
		trans_table_C.syncDeviceMemory();
		trans_table_Y.syncDeviceMemory();

		//----------------------------------------------------------------------------
		// メモリ確保
		//============================================================================
		DeviceMemory<byte> src(width * height * 3);
		DeviceMemory<int> yuv_buffer(width * height + 2 * (width / 2) * (height / 2));
		DeviceMemory<int> quantized(width * height + 2 * (width / 2) * (height / 2));

		DeviceMemory<int> dct_coeficient(width * height + 2 * (width / 2) * (height / 2));
		DeviceMemory<float> dct_tmp_buffer(width * height + 2 * (width / 2) * (height / 2));

		GPUOutBitStreamBuffer out_bit_stream_buffer((YCC_size / 64) * MBS);
		CudaMemory<GPUOutBitStreamState> out_bit_stream_status(YCC_size / 64);
		out_bit_stream_status.syncDeviceMemory();

		std::cout << "	-----------------------------------------------" << std::endl;
		std::cout << "	 Encode" << std::endl;
		std::cout << "	-----------------------------------------------" << std::endl;
		watch.start();
		{
			//----------------------------------------------------------------------------
			// メモリ転送
			//============================================================================
			src.write_device((byte*) source.getRawData(), width * height * 3);

			//----------------------------------------------------------------------------
			// RGB->yuv
			//============================================================================
			gpu_color_trans_Y<<<Dg0_0, Db0_0>>>(src.device_data(), yuv_buffer.device_data(), trans_table_Y.device_data());
			gpu_color_trans_C<<<Dg0_0, Db0_0>>>(src.device_data(), yuv_buffer.device_data(), trans_table_C.device_data(), height, C_size);

			//----------------------------------------------------------------------------
			// DCT
			//============================================================================
			gpu_dct_0<<<Dg1, Db1>>>(yuv_buffer.device_data(), dct_tmp_buffer.device_data());
			gpu_dct_1<<<Dg1, Db1>>>(dct_tmp_buffer.device_data(), dct_coeficient.device_data());

			//----------------------------------------------------------------------------
			// 量子化
			//============================================================================
			gpu_zig_quantize_Y<<<Dg2_0, Db2_0>>>(dct_coeficient.device_data(), quantized.device_data());
			gpu_zig_quantize_C<<<Dg2_1, Db2_1>>>(dct_coeficient.device_data(), quantized.device_data(), Y_size);
			//----------------------------------------------------------------------------
			// ハフマン符号化
			//============================================================================
			gpu_huffman_mcu<<<Dg3_0, Db3_0>>>(quantized.device_data(), out_bit_stream_status.device_data(),
				out_bit_stream_buffer.getWriteBufAddress(), out_bit_stream_buffer.getEndOfBuf(), width, height);

			// 逐次処理のためCPUに戻す
			out_bit_stream_status.syncHostMemory();
			cpu_huffman_middle(out_bit_stream_status.host_data(), width, height, num_bits.data());
			out_bit_stream_status.syncDeviceMemory();

			gpu_huffman_write_devide0<<<Dg3_1, Db3_1>>>(out_bit_stream_status.device_data(),
				out_bit_stream_buffer.getWriteBufAddress(), encode_result.device_data(), width, height);
			gpu_huffman_write_devide1<<<Dg3_1, Db3_1>>>(out_bit_stream_status.device_data(),
				out_bit_stream_buffer.getWriteBufAddress(), encode_result.device_data(), width, height);
			gpu_huffman_write_devide2<<<Dg3_1, Db3_1>>>(out_bit_stream_status.device_data(),
				out_bit_stream_buffer.getWriteBufAddress(), encode_result.device_data(), width, height);

			//----------------------------------------------------------------------------
			// 結果メモリ転送 :出力は「dst_dataとdst_NumBits」の２つ
			//============================================================================
			result_size = out_bit_stream_status[YCC_size / 64 - 1].mBytePos
				+ (out_bit_stream_status[YCC_size / 64 - 1].mBitPos == 7 ? 0 : 1);
			encode_result.syncHostMemory();
		}
		watch.lap();
		watch.stop();
		std::cout << "	" << watch.getLastElapsedTime() << "[ms]\n" << std::endl;
	}

	BitmapCVUtil result(width, height, 8, source.getBytePerPixel());
	{
		//----------------------------------------------------------------------------
		// 色変換テーブルの作成 Decode用定数転送
		//============================================================================
		CudaMemory<int> itrans_table_Y(width * height);
		CudaMemory<int> itrans_table_C(width * height);
		make_itrans_table(itrans_table_Y.host_data(), itrans_table_C.host_data(), width, height);
		itrans_table_C.syncDeviceMemory();
		itrans_table_Y.syncDeviceMemory();

		//----------------------------------------------------------------------------
		// メモリ確保
		//============================================================================
		DeviceMemory<byte> src(width * height * 3);
		DeviceMemory<int> yuv_buffer(width * height + 2 * (width / 2) * (height / 2));
		DeviceMemory<int> quantized(width * height + 2 * (width / 2) * (height / 2));

		DeviceMemory<int> dct_coeficient(width * height + 2 * (width / 2) * (height / 2));
		DeviceMemory<float> dct_tmp_buffer(width * height + 2 * (width / 2) * (height / 2));

		std::cout << "	-----------------------------------------------" << std::endl;
		std::cout << "	 Decode" << std::endl;
		std::cout << "	-----------------------------------------------" << std::endl;
		watch.start();
		{
			//----------------------------------------------------------------------------
			// メモリ確保
			//============================================================================
			InBitStream mIBSP(encode_result.host_data(), result_size);
			IntBuffer c_qua(width * height + 2 * (width / 2) * (height / 2));

			//----------------------------------------------------------------------------
			// ハフマン復号
			//============================================================================
			// CPU
			decode_huffman(&mIBSP, c_qua.data(), width, height);
			// GPU:GPUInstream.hにバグがある可能性もあるので留意
			cudaMemcpy(quantized.device_data(), c_qua.data(),
				sizeof(int) * (width * height + 2 * (width / 2) * (height / 2)),
				cudaMemcpyHostToDevice);

			//----------------------------------------------------------------------------
			// 逆量子化
			//============================================================================
			gpu_izig_quantize_Y<<<Dg2_0, Db2_0>>>(quantized.device_data(), dct_coeficient.device_data());
			gpu_izig_quantize_C<<<Dg2_1, Db2_1>>>(quantized.device_data(), dct_coeficient.device_data(), Y_size);

			//----------------------------------------------------------------------------
			// 逆DCT
			//============================================================================
			gpu_idct_0<<<Dg1, Db1>>>(dct_coeficient.device_data(), dct_tmp_buffer.device_data());
			gpu_idct_1<<<Dg1, Db1>>>(dct_tmp_buffer.device_data(), yuv_buffer.device_data());

			//----------------------------------------------------------------------------
			// yuv->RGB
			//============================================================================
			gpu_color_itrans<<<Dg0_0, Db0_0>>>(yuv_buffer.device_data(), src.device_data(),
				itrans_table_Y.device_data(), itrans_table_C.device_data(), C_size);

			//----------------------------------------------------------------------------
			// 結果転送
			//============================================================================
			cudaMemcpy((byte*) result.getRawData(), src.device_data(), src.size(),
				cudaMemcpyDeviceToHost);
		}
		watch.lap();
		watch.stop();
		std::cout << "	" << watch.getLastElapsedTime() << "[ms]\n" << std::endl;
	}
	result.saveToFile("gpu_" + out_file_name);

	std::cout << "-----------------------------------------------" << std::endl;
	std::cout << " Finish GPU Encoding & Decoding" << std::endl;
	std::cout << "===============================================\n\n" << std::endl;
}

