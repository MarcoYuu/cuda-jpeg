#include <cstdlib>
#include <cstring>

#include <string>
#include <fstream>
#include <iostream>

#include "cpu_jpeg.h"

#include "utils/timer.h"
#include "utils/util_cv.h"
#include "utils/encoder_tables.h"

using namespace std;
using namespace util;
using namespace jpeg;

//----------------------------------------------------------------------------
// CPU Jpeg圧縮テストルーチン
//============================================================================
void cpu_exec(const std::string &file_name, const std::string &out_file_name) {
	StopWatch watch(StopWatch::CPU_OPTIMUM);

	// 画像データを読み出し
	BitmapCVUtil source(file_name, BitmapCVUtil::RGB_COLOR);
	const int width = source.getWidth();
	const int height = source.getHeight();

	std::cout << "===============================================" << std::endl;
	std::cout << " Start CPU Encoding & Decoding" << std::endl;
	std::cout << "-----------------------------------------------\n" << std::endl;

	ByteBuffer encode_result;
	{
		std::cout << "	-----------------------------------------------" << std::endl;
		std::cout << "	 Encode" << std::endl;
		std::cout << "	-----------------------------------------------" << std::endl;

		JpegEncoder encoder(width, height);

		watch.start();
		size_t data_size = encoder.encode((byte*) source.getRawData(), width * height * 3,
			encode_result);
		watch.lap();
		watch.stop();
		std::cout << "	" << watch.getLastElapsedTime() * 1000 << "[ms]\n" << std::endl;
	}

	BitmapCVUtil result(width, height, 8, source.getBytePerPixel());
	{
		std::cout << "	-----------------------------------------------" << std::endl;
		std::cout << "	 Decode" << std::endl;
		std::cout << "	-----------------------------------------------" << std::endl;

		JpegDecoder decoder(width, height);

		watch.start();
		decoder.decode(encode_result.data(), encode_result.size(), (byte*) result.getRawData(),
			width * height * 3);
		watch.lap();
		watch.stop();
		std::cout << "	" << watch.getLastElapsedTime() * 1000 << "[ms]\n" << std::endl;
	}

	result.saveToFile("cpu_decoder_" + out_file_name);

	std::cout << "-----------------------------------------------" << std::endl;
	std::cout << " Finish CPU Encoding & Decoding" << std::endl;
	std::cout << "===============================================\n\n" << std::endl;
}
