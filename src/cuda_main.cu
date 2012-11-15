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
	device_memory<byte> src(width * height * 3);
	src.write_device((byte*) source.getRawData(), width * height * 3);

	// ブロックサイズ
	const int BLOCK_SIZE = 64;

	{
		// 色変換
		cuda_memory<byte> result(width * height * 3 / 2);
		cuda_memory<int> table_result(width * height * 3 / 2);

		ConvertRGBToYUV(src, result, width, height, BLOCK_SIZE, BLOCK_SIZE, table_result);
		result.sync_to_host();
		table_result.sync_to_host();

		dim3 grid(BLOCK_SIZE / 16, BLOCK_SIZE / 16, width / BLOCK_SIZE * height / BLOCK_SIZE);
		dim3 block(16, 16, 1);

		// 結果の出力
		{
			std::ofstream ofs("table.txt");
			ofs << "dst, src" << endl;
			for (int i = 0; i < table_result.size(); ++i) {
				ofs << i << "," << table_result[i] << endl;
			}
			ofs.close();

			ofs.open("yuv.txt");
			for (int i = 0; i < result.size(); ++i) {
				ofs << static_cast<int>(result[i]) << endl;
			}
			ofs.close();
		}
	}

	std::cout << "-----------------------------------------------" << std::endl;
	std::cout << " Finish GPU Encoding & Decoding" << std::endl;
	std::cout << "===============================================\n\n" << std::endl;
}

