/*
 * cuda_main.cpp
 *
 *  Created on: 2012/11/12
 *      Author: momma
 */

#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

#include <boost/lexical_cast.hpp>

#include "jpeg/cpu/cpu_jpeg.h"
#include "jpeg/cuda/cuda_jpeg.cuh"

#include "utils/debug_log.h"
#include "utils/util_cv.h"
#include "utils/type_definitions.h"

#include "utils/cuda/cuda_timer.h"
#include "utils/cuda/cuda_memory.hpp"

using namespace std;
using namespace util;
using namespace util::cuda;
using namespace jpeg;
using namespace jpeg::cuda;

struct TableExport: public util::DebugLog::OutputFormat {
private:
	const size_t size_;
	jpeg::cuda::CudaTable &table_;

public:
	TableExport(jpeg::cuda::CudaTable &yuv, int block_size) :
		size_(block_size),
		table_(yuv) {
	}
	void operator()(std::ofstream& ofs) const {
		table_.sync_to_host();
		ofs << table_.size() << std::endl;
		for (int i = 0; i < table_.size(); ++i) {
			if (i % size_ == 0)
				ofs << "\n\n\nblock :" << i / size_;

			if (i % 8 == 0)
				ofs << std::endl;

			if (i % 64 == 0)
				ofs << "\n8x8block :" << i << std::endl;

			//ofs << "(" << i << " -> " << table_[i].y << ", " << table_[i].u << ", " << table_[i].v << "), ";
			ofs << "(" << i << ", " << table_[i].y << "), ";
		}
	}
};

/**
 * @brief ログファイル出力用
 *
 * @author yuumomma
 * @version 1.0
 */
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

void CalcurateMatrixTest();

void encoder_decoder(const std::string& file_name, const std::string& out_file_name, size_t block_width,
	size_t block_height, int quarity);

void encode_and_decode(const std::string& file_name, const std::string& out_file_name, size_t block_width,
	size_t block_height, int quarity);

void color_conversion_only(const std::string &file_name, const std::string &out_file_name, size_t block_width,
	size_t block_height, int quarity);

void code_func(const std::string &file_name, const std::string &out_file_name, size_t block_width,
	size_t block_height, int quarity);

void cuda_main(const std::string &file_name, const std::string &out_file_name, size_t block_width,
	size_t block_height, int quarity) {
	code_func(file_name, out_file_name, block_width, block_height, quarity);
	//encoder_decoder(file_name, out_file_name, block_width, block_height, quarity);
	//encode_and_decode(file_name, out_file_name, block_width, block_height, quarity);
	//color_conversion_only(file_name, out_file_name, block_width, block_height, quarity);
}

void encode(const byte* rgb, byte* huffman, size_t width, size_t height, size_t block_width,
	size_t block_height, int quarity) {

	// ブロックサイズ
	const int IMG_MEM_SIZE = width * height * 3 / 2;
	const int BLOCK_WIDTH = block_width == 0 ? width : block_width;
	const int BLOCK_HEIGHT = block_height == 0 ? height : block_height;
	const int BLOCK_MEM_SIZE = BLOCK_WIDTH * BLOCK_HEIGHT * 3 / 2;
	const int BLOCK_NUM = width * height / (BLOCK_WIDTH * BLOCK_HEIGHT);

	DeviceByteBuffer encode_src(rgb, width * height * 3);
	CudaTable encode_table(width * height);
	CudaByteBuffer encode_yuv_result(IMG_MEM_SIZE);
	CudaIntBuffer encode_dct_result(IMG_MEM_SIZE);
	CudaIntBuffer encode_qua_result(IMG_MEM_SIZE);
	CudaByteBuffer encode_huffman_result(IMG_MEM_SIZE);
	IntBuffer huffman_effective_bits(BLOCK_NUM);

	CreateConversionTable(width, height, BLOCK_WIDTH, BLOCK_HEIGHT, encode_table);
	ConvertRGBToYUV(encode_src, encode_yuv_result, width, height, BLOCK_WIDTH, BLOCK_HEIGHT, encode_table);
	DiscreteCosineTransform(encode_yuv_result, encode_dct_result);
	ZigzagQuantize(encode_dct_result, encode_qua_result, BLOCK_MEM_SIZE, quarity);

	encode_huffman_result.fill_zero();
	HuffmanEncode(encode_qua_result, encode_huffman_result, huffman_effective_bits);
	encode_huffman_result.copy_to_host(huffman, IMG_MEM_SIZE);
}

void decode(const byte *huffman, byte *dst, size_t width, size_t height, size_t block_width,
	size_t block_height, int quarity) {

	// ブロックサイズ
	const int IMG_MEM_SIZE = width * height * 3 / 2;
	const int BLOCK_WIDTH = block_width == 0 ? width : block_width;
	const int BLOCK_HEIGHT = block_height == 0 ? height : block_height;
	const int BLOCK_MEM_SIZE = BLOCK_WIDTH * BLOCK_HEIGHT * 3 / 2;

	CudaTable decode_table(BLOCK_WIDTH * BLOCK_HEIGHT);
	CudaByteBuffer decode_huffman_src(BLOCK_MEM_SIZE);
	CudaIntBuffer decode_qua_src(BLOCK_MEM_SIZE);
	CudaIntBuffer decode_dct_src(BLOCK_MEM_SIZE);
	CudaByteBuffer decode_yuv_src(BLOCK_MEM_SIZE);
	CudaByteBuffer decode_result(BLOCK_WIDTH * BLOCK_HEIGHT * 3);

	CreateConversionTable(BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT, decode_table);

	InBitStream ibs(huffman, IMG_MEM_SIZE);
	cpu::decode_huffman(&ibs, decode_qua_src.host_data(), BLOCK_WIDTH, BLOCK_HEIGHT);
	decode_qua_src.sync_to_device();

	InverseZigzagQuantize(decode_qua_src, decode_dct_src, BLOCK_MEM_SIZE, quarity);

	InverseDiscreteCosineTransform(decode_dct_src, decode_yuv_src);
	decode_yuv_src.sync_to_host();

	ConvertYUVToRGB(decode_yuv_src, decode_result, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT,
		decode_table);
	decode_result.copy_to_host(dst, decode_result.size());
}

void code_func(const std::string &file_name, const std::string &out_file_name, size_t block_width,
	size_t block_height, int quarity) {

	// 画像読み込み
	BitmapCVUtil source(file_name, BitmapCVUtil::RGB_COLOR);
	const int width = source.getWidth();
	const int height = source.getHeight();

	ByteBuffer huffman(width * height * 3 / 2);
	encode((byte*) source.getRawData(), huffman.data(), width, height, block_width, block_height, quarity);

	const int BLOCK_NUM = width * height / (block_width * block_height);
	BitmapCVUtil bmp(block_width, block_height, 8, source.getBytePerPixel());
	for (int i = 0; i < BLOCK_NUM; ++i) {
		decode(huffman.data() + huffman.size() / BLOCK_NUM * i, (byte*) bmp.getRawData(), block_width,
			block_height, block_width, block_height, quarity);

		string index = boost::lexical_cast<string>(i);
		string qrty = boost::lexical_cast<string>(quarity);
		string outname = "cuda_" + index + "_" + qrty + "_" + out_file_name;
		DebugLog::log("export to file :" + outname);
		bmp.saveToFile(outname);
	}
}

void encoder_decoder(const std::string& file_name, const std::string& out_file_name, size_t block_width,
	size_t block_height, int quarity) {

	using namespace std;
	using namespace util;
	using namespace util::cuda;
	using namespace jpeg;
	using namespace jpeg::cuda;

	DebugLog::initTimer(true);

	// 画像読み込み
	BitmapCVUtil source(file_name, BitmapCVUtil::RGB_COLOR);
	const int width = source.getWidth();
	const int height = source.getHeight();
	const int block_num = width * height / (block_width * block_height);

	DebugLog::startSection("Start CUDA Encoding & Decoding");

	DeviceByteBuffer encode_src(width * height * 3);
	CudaByteBuffer encode_result(width * height * 3 / 2);
	IntBuffer effective_bits(block_num);
	Encoder encoder(width, height, block_width, block_height);
	encoder.setQuarity(quarity);
	{
		DebugLog::startLoggingTime("Encode");

		encode_src.write_device((byte*) ((source.getRawData())), encode_src.size());
		encoder.encode(encode_src, encode_result, effective_bits);
		encode_result.sync_to_host();

		DebugLog::endLoggingTime();
	}

	BitmapCVUtil bmp(block_width, block_height, 8, source.getBytePerPixel());
	CudaByteBuffer decode_result(block_width * block_height * 3);
	Decoder decoder(block_width, block_height);
	decoder.setQuarity(quarity);
	{
		for (int i = 0; i < block_num; ++i) {
			DebugLog::startLoggingTime("Decode");

			decoder.decode(encode_result.host_data(), encode_result.size() / block_num, decode_result);
			decode_result.sync_to_host();

			DebugLog::endLoggingTime();

			string index = boost::lexical_cast<string>(i);
			string qrty = boost::lexical_cast<string>(quarity);
			string outname = "cuda_" + index + "_" + qrty + "_" + out_file_name;
			decode_result.copy_to_host((byte*) ((bmp.getRawData())), decode_result.size());
			bmp.saveToFile(outname);
			DebugLog::log("export to file :" + outname);
		}
	}

	DebugLog::endSection("Finish CUDA Encoding & Decoding");
}

void encode_and_decode(const std::string& file_name, const std::string& out_file_name, size_t block_width,
	size_t block_height, int quarity) {

	DebugLog::initTimer(true);

	// 画像読み込み
	BitmapCVUtil source(file_name, BitmapCVUtil::RGB_COLOR);
	const int width = source.getWidth();
	const int height = source.getHeight();

	// ブロックサイズ
	const int IMG_MEM_SIZE = width * height * 3 / 2;
	const int BLOCK_WIDTH = block_width == 0 ? width : block_width;
	const int BLOCK_HEIGHT = block_height == 0 ? height : block_height;
	const int BLOCK_MEM_SIZE = BLOCK_WIDTH * BLOCK_HEIGHT * 3 / 2;
	const int BLOCK_NUM = width * height / (BLOCK_WIDTH * BLOCK_HEIGHT);

	DebugLog::startSection("Start CUDA Encoding & Decoding");

	DebugLog::log("infile_name: " + file_name);
	DebugLog::log("outfile_name: " + out_file_name);
	DebugLog::log("block_width: " + boost::lexical_cast<string>(block_width));
	DebugLog::log("block_height: " + boost::lexical_cast<string>(block_height));
	DebugLog::log("quarity: " + boost::lexical_cast<string>(quarity));

	DeviceByteBuffer encode_src((byte*) ((source.getRawData())), width * height * 3);
	CudaTable encode_table(width * height);
	CudaByteBuffer encode_yuv_result(IMG_MEM_SIZE);
	CudaIntBuffer encode_dct_result(IMG_MEM_SIZE);
	CudaIntBuffer encode_qua_result(IMG_MEM_SIZE);
	CudaByteBuffer encode_huffman_result(IMG_MEM_SIZE);
	IntBuffer huffman_effective_bits(BLOCK_NUM);
	{
		DebugLog::startSubSection("Encode");
		DebugLog::startLoggingTime("CreateConvertTable");
		CreateConversionTable(width, height, BLOCK_WIDTH, BLOCK_HEIGHT, encode_table);
		DebugLog::endLoggingTime();

		DebugLog::startLoggingTime("ConvertRGBToYUV");
		ConvertRGBToYUV(encode_src, encode_yuv_result, width, height, BLOCK_WIDTH, BLOCK_HEIGHT,
			encode_table);
		DebugLog::endLoggingTime();
		DebugLog::exportToFile("encode_yuv_result.txt",
			BlockExport<CudaByteBuffer>(encode_yuv_result, BLOCK_MEM_SIZE));

		DebugLog::startLoggingTime("DiscreteCosineTransform");
		DiscreteCosineTransform(encode_yuv_result, encode_dct_result);
		DebugLog::endLoggingTime();
		DebugLog::exportToFile("encode_dct_result.txt",
			BlockExport<CudaIntBuffer>(encode_dct_result, BLOCK_MEM_SIZE));

		DebugLog::startLoggingTime("ZigzagQuantize");
		ZigzagQuantize(encode_dct_result, encode_qua_result, BLOCK_MEM_SIZE, quarity);
		DebugLog::endLoggingTime();
		DebugLog::exportToFile("encode_qua_result.txt",
			BlockExport<CudaIntBuffer>(encode_qua_result, BLOCK_MEM_SIZE));

		DebugLog::startLoggingTime("HuffmanEncode");
		encode_huffman_result.fill_zero();
		HuffmanEncode(encode_qua_result, encode_huffman_result, huffman_effective_bits);
		encode_huffman_result.sync_to_host();
		DebugLog::endLoggingTime();

		DebugLog::dump_memory(encode_huffman_result.host_data(), huffman_effective_bits[0] / 8 + 1,
			"huffman.finish.dat");

		DebugLog::printTotalTime();
		DebugLog::endSubSection();
	}

	DebugLog::resetTotalTime();
	BitmapCVUtil bmp(BLOCK_WIDTH, BLOCK_HEIGHT, 8, source.getBytePerPixel());

	CudaTable decode_table(BLOCK_WIDTH * BLOCK_HEIGHT);
	CudaByteBuffer decode_huffman_src(BLOCK_MEM_SIZE);
	CudaIntBuffer decode_qua_src(BLOCK_MEM_SIZE);
	CudaIntBuffer decode_dct_src(BLOCK_MEM_SIZE);
	CudaByteBuffer decode_yuv_src(BLOCK_MEM_SIZE);
	CudaByteBuffer decode_result(BLOCK_WIDTH * BLOCK_HEIGHT * 3);
	{
		DebugLog::startSubSection("Decode");

		DebugLog::startLoggingTime("CreateConvertTable");
		CreateConversionTable(BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT, decode_table);
		DebugLog::endLoggingTime();

		for (int i = 0; i < width / BLOCK_WIDTH * height / BLOCK_HEIGHT; ++i) {
			string index = boost::lexical_cast<string>(i);

			InBitStream ibs(encode_huffman_result.host_data() + encode_huffman_result.size() / BLOCK_NUM * i,
				encode_huffman_result.size() / BLOCK_NUM);
			cpu::decode_huffman(&ibs, decode_qua_src.host_data(), BLOCK_WIDTH, BLOCK_HEIGHT);
			decode_qua_src.sync_to_device();
			DebugLog::exportToFile("decode_qua_src" + index + ".txt",
				BlockExport<CudaIntBuffer>(decode_qua_src, BLOCK_MEM_SIZE));

			DebugLog::startLoggingTime("InverseZigzagQuantize");
			InverseZigzagQuantize(decode_qua_src, decode_dct_src, BLOCK_MEM_SIZE, quarity);
			DebugLog::endLoggingTime();
			DebugLog::exportToFile("decode_dct_src" + index + ".txt",
				BlockExport<CudaIntBuffer>(decode_dct_src, BLOCK_MEM_SIZE));

			DebugLog::startLoggingTime("InverseDiscreteCosineTransform");
			InverseDiscreteCosineTransform(decode_dct_src, decode_yuv_src);
			DebugLog::endLoggingTime();
			DebugLog::exportToFile("decode_yuv_src" + index + ".txt",
				BlockExport<CudaByteBuffer>(decode_yuv_src, BLOCK_MEM_SIZE));
			decode_yuv_src.sync_to_host();

			DebugLog::startLoggingTime("ConvertYUVToRGB");
			ConvertYUVToRGB(decode_yuv_src, decode_result, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_WIDTH,
				BLOCK_HEIGHT, decode_table);
			decode_result.copy_to_host((byte*) ((bmp.getRawData())), decode_result.size());
			DebugLog::endLoggingTime();

			string qrty = boost::lexical_cast<string>(quarity);
			string outname = "cuda_" + index + "_" + qrty + "_" + out_file_name;
			DebugLog::log("export to file :" + outname);
			bmp.saveToFile(outname);
		}
		DebugLog::printTotalTime();
		DebugLog::endSubSection();
	}
	DebugLog::endSection("Finish CUDA Encoding & Decoding");
}

void color_conversion_only(const std::string &file_name, const std::string &out_file_name, size_t block_width,
	size_t block_height, int quarity) {

	// 画像読み込み
	BitmapCVUtil source(file_name, BitmapCVUtil::RGB_COLOR);
	const int width = source.getWidth();
	const int height = source.getHeight();

	// ブロックサイズ
	const int BLOCK_WIDTH = block_width == 0 ? width : block_width;
	const int BLOCK_HEIGHT = block_height == 0 ? height : block_height;
	const int BLOCK_MEM_SIZE = BLOCK_WIDTH * BLOCK_HEIGHT * 3 / 2;
	const int IMG_MEM_SIZE = width * height * 3 / 2;

	DebugLog::startSection("Start CUDA Encoding & Decoding");

	DeviceByteBuffer encode_src((byte*) (source.getRawData()), width * height * 3);
	CudaTable encode_table(width * height);
	CudaByteBuffer encode_yuv_result(IMG_MEM_SIZE);
	{
		DebugLog::startSubSection("Encode");

		CreateConversionTable(width, height, BLOCK_WIDTH, BLOCK_HEIGHT, encode_table);
		DebugLog::exportToFile("encode_table.txt", TableExport(encode_table, BLOCK_WIDTH * BLOCK_HEIGHT));

		ConvertRGBToYUV(encode_src, encode_yuv_result, width, height, BLOCK_WIDTH, BLOCK_HEIGHT,
			encode_table);
		encode_yuv_result.sync_to_host();
		DebugLog::exportToFile("encode_yuv_result.txt",
			BlockExport<CudaByteBuffer>(encode_yuv_result, BLOCK_MEM_SIZE));

		DebugLog::endSubSection();
	}

	BitmapCVUtil bmp(BLOCK_WIDTH, BLOCK_HEIGHT, 8, source.getBytePerPixel());
	CudaTable decode_table(BLOCK_WIDTH * BLOCK_HEIGHT);
	CudaByteBuffer decode_yuv_src(BLOCK_MEM_SIZE);
	CudaByteBuffer decode_result(BLOCK_WIDTH * BLOCK_HEIGHT * 3);
	{
		DebugLog::startSubSection("Decode");

		CreateConversionTable(BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT, decode_table);
		DebugLog::exportToFile("decode_table.txt", TableExport(decode_table, BLOCK_WIDTH * BLOCK_HEIGHT));

		for (int i = 0; i < width / BLOCK_WIDTH * height / BLOCK_HEIGHT; ++i) {
			string index = boost::lexical_cast<string>(i);

			DebugLog::log("copy block memory.");
			decode_yuv_src.write_device(encode_yuv_result.host_data() + i * decode_yuv_src.size(),
				decode_yuv_src.size());

			ConvertYUVToRGB(decode_yuv_src, decode_result, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_WIDTH,
				BLOCK_HEIGHT, decode_table);
			decode_result.copy_to_host((byte*) (bmp.getRawData()), decode_result.size());
			DebugLog::exportToFile("decode_yuv_src" + index + ".txt",
				BlockExport<CudaByteBuffer>(decode_yuv_src, BLOCK_MEM_SIZE));
			decode_yuv_src.sync_to_host();

			string outname = "cudacolor_" + index + "_" + boost::lexical_cast<string>(quarity) + "_"
				+ out_file_name;

			DebugLog::log("export to file :" + outname);

			bmp.saveToFile(outname);
		}

		DebugLog::endSubSection();
	}

	DebugLog::endSection("");
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

