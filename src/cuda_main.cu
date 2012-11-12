/*
 * cuda_main.cpp
 *
 *  Created on: 2012/11/12
 *      Author: momma
 */

#include <cstdlib>
#include <cstring>
#include <cmath>

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

#include "cpu_jpeg.h"
#include "gpu_jpeg.cuh"
#include "cuda_jpeg.cuh"

#include "utils/utils.hpp"
#include "utils/util_cv.h"
#include "utils/timer.h"
#include "utils/cuda_timer.h"
#include "utils/cuda_memory.hpp"
#include "utils/util_cv.h"
#include "utils/encoder_tables.h"
#include "utils/in_bit_stream.h"

void cuda_exec(const std::string &file_name, const std::string &out_file_name) {
	using namespace std;
	using namespace util;
	using namespace util::cuda;
	using namespace jpeg;
	using namespace jpeg::cuda;

	//----------------------------------------------------------------------------
	// 画像読み込み
	//============================================================================
	BitmapCVUtil source(file_name, BitmapCVUtil::RGB_COLOR);
	const int width = source.getWidth();
	const int height = source.getHeight();
	device_memory<byte> src(width * height * 3 / 2);
	src.write_device((byte*) source.getRawData(), width * height * 3);
	std::ofstream data("src.txt");
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			data << "("
				<< (int) (((byte*) source.getRawData())[(y * width + x) * 3 + 0]) << ","
				<< (int) (((byte*) source.getRawData())[(y * width + x) * 3 + 1]) << ","
				<< (int) (((byte*) source.getRawData())[(y * width + x) * 3 + 2]) <<
				"), ";
		}
		data << endl;
	}

	// ブロックサイズ
	const int BLOCK_SIZE = 64;

	{
		// 色変換
		cuda_memory<int> result(width * height * 3 / 2);
		result.fill_zero();
		result.sync_to_device();
		dim3 grid(BLOCK_SIZE / 16, BLOCK_SIZE / 16, width / BLOCK_SIZE * height / BLOCK_SIZE);
		dim3 block(16, 16, 1);
		ConvertRGBToYUV<<<grid,block>>>(
			src.device_data(), result.device_data(), width, height, BLOCK_SIZE, BLOCK_SIZE);

		result.sync_to_host();
		dump_memory(result.host_data(), result.size(), "momma.dat");
		std::ofstream data("momma.txt");
		data << "Y:{" << endl;
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				data << result[y * width + x] << ",";
			}
			data << endl;
		}
		data << "}\n" << endl;
		data << "U:{" << endl;
		for (int y = 0; y < height / 2; ++y) {
			for (int x = 0; x < width / 2; ++x) {
				data << result[width * height + y * width + x] << ",";
			}
			data << endl;
		}
		data << "}\n" << endl;
		data << "V:{" << endl;
		for (int y = 0; y < height / 2; ++y) {
			for (int x = 0; x < width / 2; ++x) {
				data << result[width * height * 5 / 4 + y * width + x] << ",";
			}
			data << endl;
		}
		data << "}\n" << endl;

		// 結果バッファ
		BitmapCVUtil dest(BLOCK_SIZE, BLOCK_SIZE, 8, 3);
		cuda_memory<byte> result_decode(BLOCK_SIZE * BLOCK_SIZE * 3);
		cuda_memory<int> table_y(BLOCK_SIZE * BLOCK_SIZE);
		cuda_memory<int> table_c(BLOCK_SIZE * BLOCK_SIZE);

		// 変換テーブルの作成
		make_itrans_table(table_y.host_data(), table_c.host_data(), BLOCK_SIZE, BLOCK_SIZE);
		table_y.sync_to_device();
		table_c.sync_to_device();

		// 色変化案
		dim3 grid_i(BLOCK_SIZE * BLOCK_SIZE / 256, 1, 1);
		dim3 block_i(256, 1, 1);
		gpu_color_itrans<<<grid_i,block_i>>>(
			result.device_data(), result_decode.device_data(), table_y.device_data(), table_c.device_data(),
			width * height / 4);
		result_decode.copy_to_host((byte*) dest.getRawData(), result_decode.size());
		dest.saveToFile("cconv_" + out_file_name);
	}

	std::cout << "-----------------------------------------------" << std::endl;
	std::cout << " Finish GPU Encoding & Decoding" << std::endl;
	std::cout << "===============================================\n\n" << std::endl;
}

