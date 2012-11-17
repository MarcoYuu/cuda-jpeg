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

#include "jpeg/cpu/cpu_jpeg.h"
#include "jpeg/ohmura/gpu_jpeg.cuh"
#include "jpeg/cuda/cuda_jpeg.cuh"

#include "utils/utils.hpp"
#include "utils/util_cv.h"
#include "utils/timer.h"

#include "utils/cuda/cuda_timer.h"
#include "utils/cuda/cuda_memory.hpp"

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

	{
		ofstream ofs("source.txt");
		for (int i = 0; i < width * height * 3; ++i) {
			ofs << "B," << (int) ((byte*) source.getRawData())[i];
			ofs << ",G," << (int) ((byte*) source.getRawData())[i + 1];
			ofs << ",R," << (int) ((byte*) source.getRawData())[i + 2] << endl;
		}
	}

	// ブロックサイズ
	const int BLOCK_WIDTH = 32;
	const int BLOCK_HEIGHT = 32;

	std::cout << "===============================================" << std::endl;
	std::cout << " Start CUDA Encoding & Decoding" << std::endl;
	std::cout << "-----------------------------------------------\n" << std::endl;

	cuda_memory<byte> encode_result(width * height * 3 / 2);
	encode_result.fill_zero();
	{
		std::cout << "	-----------------------------------------------" << std::endl;
		std::cout << "	 Encode" << std::endl;
		std::cout << "	-----------------------------------------------\n" << std::endl;

		std::cout << "	 	CreateConvertTable" << std::endl;
		cuda_memory<int> table(width * height);
		table.fill_zero();
		CreateConvertTable(width, height, BLOCK_WIDTH, BLOCK_HEIGHT, table);

		{
			table.sync_to_host();
			ofstream ofs("table.txt");
			for (int i = 0; i < table.size(); ++i) {
				ofs << i << "," << table[i] / 3 << endl;
			}
		}

		std::cout << "	 	ConvertRGBToYUV" << std::endl;
		ConvertRGBToYUV(src, encode_result, width, height, BLOCK_WIDTH, BLOCK_HEIGHT, table);

		{
			encode_result.sync_to_host();
			ofstream ofs("encode_result.txt");
			for (int i = 0; i < encode_result.size(); ++i) {
				if (i < width * height) {
					ofs << "Y," << i << "," << (int) encode_result[i] << endl;
				} else if (i < width * height * 5 / 4) {
					ofs << "U," << i - width * height << "," << (int) encode_result[i] << endl;
				} else {
					ofs << "V," << i - width * height * 5 / 4 << "," << (int) encode_result[i]
						<< endl;
				}
			}
		}
	}

	cuda_memory<byte> decode_result(BLOCK_WIDTH * BLOCK_HEIGHT * 3);
	decode_result.fill_zero();
	{
		std::cout << "	-----------------------------------------------" << std::endl;
		std::cout << "	 Decode" << std::endl;
		std::cout << "	-----------------------------------------------\n" << std::endl;

		std::cout << "	 	CreateConvertTable" << std::endl;
		cuda_memory<int> table(BLOCK_WIDTH * BLOCK_HEIGHT);
		CreateConvertTable(BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT, table);

		{
			table.sync_to_host();
			ofstream ofs("table_decode.txt");
			for (int i = 0; i < table.size(); ++i) {
				ofs << i << "," << table[i] / 3 << endl;
			}
		}

		std::cout << "	 	ConvertYUVToRGB" << std::endl;
		ConvertYUVToRGB(encode_result, decode_result, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_WIDTH,
			BLOCK_HEIGHT, table);

		BitmapCVUtil bmp(BLOCK_WIDTH, BLOCK_HEIGHT, 8, source.getBytePerPixel());
		decode_result.copy_to_host((byte*) bmp.getRawData(), decode_result.size());
		bmp.saveToFile("cuda_" + out_file_name);

		{
			ofstream ofs("decode_result.txt");
			for (int i = 0; i < BLOCK_WIDTH * BLOCK_HEIGHT * 3; ++i) {
				ofs << "B," << (int) ((byte*) bmp.getRawData())[i];
				ofs << ",G," << (int) ((byte*) bmp.getRawData())[i + 1];
				ofs << ",R," << (int) ((byte*) bmp.getRawData())[i + 2] << endl;
			}
		}
	}

	std::cout << "-----------------------------------------------" << std::endl;
	std::cout << " Finish CUDA Encoding & Decoding" << std::endl;
	std::cout << "===============================================\n\n" << std::endl;
}

