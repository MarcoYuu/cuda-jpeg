/*
 * cuda_main.cpp
 *
 *  Created on: 2012/11/12
 *      Author: momma
 */

#include <fstream>
#include <iostream>
#include <sstream>

#include <boost/lexical_cast.hpp>

#include "jpeg/cuda/cuda_jpeg.cuh"

#include "utils/debug_log.h"
#include "utils/util_cv.h"

#include "utils/cuda/cuda_timer.h"
#include "utils/cuda/cuda_memory.hpp"

template<class CudaMemory>
struct BlockExport: public util::DebugLog::OutputFormat {
private:
	const size_t size_;
	CudaMemory &yuv_;

public:
	BlockExport(CudaMemory &yuv, int block_size) :
		size_(block_size),
		yuv_(yuv) {

	}
	void operator()(std::ofstream& ofs) const {
		yuv_.sync_to_host();
		ofs << yuv_.size() << std::endl;
		for (int i = 0; i < yuv_.size(); ++i) {
			if (i % size_ == 0)
				ofs << "\n\n\nblock :" << i / size_;

			if (i % 8 == 0)
				ofs << std::endl;

			if (i % 64 == 0)
				ofs << "\n8x8block :" << i << std::endl;

			ofs << (int) (yuv_[i]) << ", ";
		}
	}
};

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

void cuda_main(const std::string &file_name, const std::string &out_file_name, size_t block_width, size_t block_height,
	int quarity) {
	using namespace std;
	using namespace util;
	using namespace util::cuda;
	using namespace jpeg;
	using namespace jpeg::cuda;

	DebugLog::init(false);

	// 画像読み込み
	BitmapCVUtil source(file_name, BitmapCVUtil::RGB_COLOR);
	const int width = source.getWidth();
	const int height = source.getHeight();

	// ブロックサイズ
	const int BLOCK_WIDTH = block_width;
	const int BLOCK_HEIGHT = block_height;
	const int BLOCK_SIZE = BLOCK_WIDTH * BLOCK_HEIGHT * 3 / 2;

	DebugLog::startSection("Start CUDA Encoding & Decoding");

	DebugLog::log("infile_name: " + file_name);
	DebugLog::log("outfile_name: " + out_file_name);
	DebugLog::log("block_width: " + boost::lexical_cast<string>(block_width));
	DebugLog::log("block_height: " + boost::lexical_cast<string>(block_height));
	DebugLog::log("quarity: " + boost::lexical_cast<string>(quarity));

	DeviceByteBuffer encode_src((byte*) (source.getRawData()), width * height * 3);
	CudaTable encode_table(width * height);
	CudaByteBuffer encode_yuv_result(width * height * 3 / 2);
	CudaIntBuffer encode_dct_result(width * height * 3 / 2);
	CudaIntBuffer encode_qua_result(width * height * 3 / 2);
	{
		DebugLog::startSubSection("Encode");

		DebugLog::startLoggingTime("CreateConvertTable");
		CreateConversionTable(width, height, BLOCK_WIDTH, BLOCK_HEIGHT, encode_table);
		DebugLog::endLoggingTime();

		DebugLog::startLoggingTime("ConvertRGBToYUV");
		ConvertRGBToYUV(encode_src, encode_yuv_result, width, height, BLOCK_WIDTH, BLOCK_HEIGHT, encode_table);
		DebugLog::endLoggingTime();
		DebugLog::exportToFile("encode_yuv_result.txt", BlockExport<CudaByteBuffer>(encode_yuv_result, BLOCK_SIZE));

		DebugLog::startLoggingTime("DiscreteCosineTransform");
		DiscreteCosineTransform(encode_yuv_result, encode_dct_result);
		DebugLog::endLoggingTime();
		DebugLog::exportToFile("encode_dct_result.txt", BlockExport<CudaIntBuffer>(encode_dct_result, BLOCK_SIZE));

		DebugLog::startLoggingTime("ZigzagQuantize");
		ZigzagQuantize(encode_dct_result, encode_qua_result, quarity);
		encode_qua_result.sync_to_host();
		DebugLog::endLoggingTime();
		DebugLog::exportToFile("encode_qua_result.txt", BlockExport<CudaIntBuffer>(encode_qua_result, BLOCK_SIZE));

		DebugLog::printTotalTime();
		DebugLog::endSubSection();
	}

	DebugLog::resetTotalTime();

	BitmapCVUtil bmp(BLOCK_WIDTH, BLOCK_HEIGHT, 8, source.getBytePerPixel());
	CudaTable decode_table(BLOCK_WIDTH * BLOCK_HEIGHT);
	CudaIntBuffer decode_qua_src(BLOCK_WIDTH * BLOCK_HEIGHT * 3 / 2);
	CudaIntBuffer decode_dct_src(BLOCK_WIDTH * BLOCK_HEIGHT * 3 / 2);
	CudaByteBuffer decode_yuv_src(BLOCK_WIDTH * BLOCK_HEIGHT * 3 / 2);
	CudaByteBuffer decode_result(BLOCK_WIDTH * BLOCK_HEIGHT * 3);
	{
		DebugLog::startSubSection("Decode");

		DebugLog::startLoggingTime("CreateConvertTable");
		CreateConversionTable(BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT, decode_table);
		DebugLog::endLoggingTime();

		for (int i = 0; i < width / BLOCK_WIDTH * height / BLOCK_HEIGHT; ++i) {
			string index = boost::lexical_cast<string>(i);

			DebugLog::log("copy block memory.");
			decode_qua_src.write_device(encode_qua_result.host_data() + i * decode_qua_src.size(), decode_qua_src.size());
			DebugLog::exportToFile("decode_qua_src" + index + ".txt", BlockExport<CudaIntBuffer>(decode_qua_src, BLOCK_SIZE));

			DebugLog::startLoggingTime("InverseZigzagQuantize");
			InverseZigzagQuantize(decode_qua_src, decode_dct_src, quarity);
			DebugLog::endLoggingTime();
			DebugLog::exportToFile("decode_dct_src" + index + ".txt", BlockExport<CudaIntBuffer>(decode_dct_src, BLOCK_SIZE));

			DebugLog::startLoggingTime("InverseDiscreteCosineTransform");
			InverseDiscreteCosineTransform(decode_dct_src, decode_yuv_src);
			DebugLog::endLoggingTime();
			DebugLog::exportToFile("decode_yuv_src" + index + ".txt", BlockExport<CudaByteBuffer>(decode_yuv_src, BLOCK_SIZE));
			decode_yuv_src.sync_to_host();

			DebugLog::startLoggingTime("ConvertYUVToRGB");
			ConvertYUVToRGB(decode_yuv_src, decode_result, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT, decode_table);
			decode_result.copy_to_host((byte*) (bmp.getRawData()), decode_result.size());
			DebugLog::endLoggingTime();

			string outname = "cuda_" + index + "_" + boost::lexical_cast<string>(quarity) + "_" + out_file_name;

			DebugLog::log("export to file :" + outname);

			bmp.saveToFile(outname);
		}
		DebugLog::printTotalTime();
		DebugLog::endSubSection();
	}
	DebugLog::endSection("Finish CUDA Encoding & Decoding");
}

