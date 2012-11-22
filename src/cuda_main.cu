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
#include <sstream>

#include "jpeg/cpu/cpu_jpeg.h"
#include "jpeg/ohmura/gpu_jpeg.cuh"
#include "jpeg/cuda/cuda_jpeg.cuh"

#include "utils/utils.hpp"
#include "utils/util_cv.h"
#include "utils/timer.h"

#include "utils/cuda/cuda_timer.h"
#include "utils/cuda/cuda_memory.hpp"

void cuda_exec(const std::string &file_name, const std::string &out_file_name, size_t block_width,
	size_t block_height) {
	using namespace std;
	using namespace util;
	using namespace util::cuda;
	using namespace jpeg;
	using namespace jpeg::cuda;

	// 画像読み込み
	BitmapCVUtil source(file_name, BitmapCVUtil::RGB_COLOR);
	const int width = source.getWidth();
	const int height = source.getHeight();

	// ブロックサイズ
	const int BLOCK_WIDTH = block_width;
	const int BLOCK_HEIGHT = block_height;
	const int BLOCK_SIZE = BLOCK_WIDTH * BLOCK_HEIGHT * 3 / 2;

	std::cout << "===============================================" << std::endl;
	std::cout << " Start CUDA Encoding & Decoding" << std::endl;
	std::cout << "-----------------------------------------------\n" << std::endl;

	DeviceByteBuffer encode_src((byte*) (source.getRawData()), width * height * 3);
	CudaByteBuffer encode_yuv_result(width * height * 3 / 2);
	CudaIntBuffer encode_dct_result(width * height * 3 / 2);
	{
		std::cout << "	-----------------------------------------------" << std::endl;
		std::cout << "	 Encode" << std::endl;
		std::cout << "	-----------------------------------------------\n" << std::endl;
		std::cout << "	 	CreateConvertTable" << std::endl;
		CudaTable table(width * height);
		CreateConversionTable(width, height, BLOCK_WIDTH, BLOCK_HEIGHT, table);
		std::cout << "	 	ConvertRGBToYUV" << std::endl;
		ConvertRGBToYUV(encode_src, encode_yuv_result, width, height, BLOCK_WIDTH, BLOCK_HEIGHT,
			table);
		encode_yuv_result.sync_to_host();
		{
			ofstream ofs("encode_yuv_result.txt");
			ofs << encode_yuv_result.size() << endl;
			for (int i = 0; i < encode_yuv_result.size(); ++i) {
				if (i % BLOCK_SIZE == 0)
					ofs << "\n\n\nblock :" << i / BLOCK_SIZE;

				if (i % 8 == 0)
					ofs << endl;

				if (i % 64 == 0)
					ofs << "\n8x8block :" << i << endl;

				ofs << (int) (encode_yuv_result[i]) << ", ";
			}
		}
		std::cout << "	 	DiscreteCosineTransform" << std::endl;
		DiscreteCosineTransform(encode_yuv_result, encode_dct_result, width, height, BLOCK_WIDTH,
			BLOCK_HEIGHT);
		encode_dct_result.sync_to_host();
		{
			ofstream ofs("encode_dct_result.txt");
			ofs << encode_dct_result.size() << endl;
			for (int i = 0; i < encode_dct_result.size(); ++i) {
				if (i % BLOCK_SIZE == 0)
					ofs << "\n\n\nblock :" << i / BLOCK_SIZE;

				if (i % 8 == 0)
					ofs << endl;

				if (i % 64 == 0)
					ofs << "\n8x8block :" << i << endl;

				ofs << (int) (encode_dct_result[i]) << ", ";
			}
		}
	}
	CudaIntBuffer decode_dct_src(BLOCK_WIDTH * BLOCK_HEIGHT * 3 / 2);
	CudaByteBuffer decode_yuv_src(BLOCK_WIDTH * BLOCK_HEIGHT * 3 / 2);
	CudaByteBuffer decode_result(BLOCK_WIDTH * BLOCK_HEIGHT * 3);
	{
		std::cout << "	-----------------------------------------------" << std::endl;
		std::cout << "	 Decode" << std::endl;
		std::cout << "	-----------------------------------------------\n" << std::endl;
		std::cout << "	 	CreateConvertTable" << std::endl;
		CudaTable table(BLOCK_WIDTH * BLOCK_HEIGHT);
		CreateConversionTable(BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT, table);
		for (int i = 0; i < width / BLOCK_WIDTH * height / BLOCK_HEIGHT; ++i) {
			std::cout << "	 	InverseDiscreteCosineTransform" << std::endl;
			decode_dct_src.write_device(encode_dct_result.host_data() + i * decode_dct_src.size(),
				decode_dct_src.size());
			decode_dct_src.sync_to_host();
			{
				stringstream s;
				s << "decode_dct_src" << i << ".txt";
				ofstream ofs(s.str().c_str());
				ofs << encode_dct_result.size() << endl;
				for (int i = 0; i < decode_dct_src.size(); ++i) {
					if (i % BLOCK_SIZE == 0)
						ofs << "\n\n\nblock :" << i / BLOCK_SIZE;

					if (i % 8 == 0)
						ofs << endl;

					if (i % 64 == 0)
						ofs << "\n8x8block :" << i << endl;

					ofs << (int) (decode_dct_src[i]) << ", ";
				}
			}
			InverseDiscreteCosineTransform(decode_dct_src, decode_yuv_src, BLOCK_WIDTH,
				BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT);
			decode_yuv_src.sync_to_host();
			{
				stringstream s;
				s << "decode_yuv_src" << i << ".txt";
				ofstream ofs(s.str().c_str());
				for (int i = 0; i < decode_yuv_src.size(); ++i) {
					if (i % BLOCK_SIZE == 0)
						ofs << "\n\n\nblock :" << i / BLOCK_SIZE;

					if (i % 8 == 0)
						ofs << endl;

					if (i % 64 == 0)
						ofs << "\n8x8block :" << i << endl;

					ofs << (int) (encode_yuv_result[i]) << ", ";
				}
			}
			std::cout << "	 	ConvertYUVToRGB" << std::endl;
			ConvertYUVToRGB(decode_yuv_src, decode_result, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_WIDTH,
				BLOCK_HEIGHT, table);
			BitmapCVUtil bmp(BLOCK_WIDTH, BLOCK_HEIGHT, 8, source.getBytePerPixel());
			decode_result.copy_to_host((byte*) (bmp.getRawData()), decode_result.size());
			stringstream s;
			s << "cuda_" << i << "_" << out_file_name;
			bmp.saveToFile(s.str());
		}

	}
	std::cout << "-----------------------------------------------" << std::endl;
	std::cout << " Finish CUDA Encoding & Decoding" << std::endl;
	std::cout << "===============================================\n\n" << std::endl;
}

void CalcurateMatrixTest() {
	using namespace jpeg::cuda;

	float DCT[64];
	CalculateDCTMatrix(DCT);
	float iDCT[64];
	CalculateiDCTMatrix(iDCT);
	for (int i = 0; i < 64; ++i) {
		if (i % 8 == 0)
			printf("\n");

		printf("%12.8f, ", DCT[i]);
	}
	std::cout << std::endl;
	for (int i = 0; i < 64; ++i) {
		if (i % 8 == 0)
			printf("\n");

		printf("%12.8f, ", iDCT[i]);
	}
}

