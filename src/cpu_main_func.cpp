#include <cstdlib>
#include <cstring>

#include <string>
#include <fstream>
#include <iostream>

using namespace std;

#include "cpu_jpeg.h"

#include "utils/timer.h"
#include "utils/util_cv.h"
#include "utils/encoder_tables.h"

//----------------------------------------------------------------------------
// コマンドライン引数解析
//============================================================================
static void parse_arg(int argc, char *argv[], string &in_file, string &out_file) {
	if (argc == 3) {
		in_file = argv[1];
		out_file = argv[2];
	} else {
		cout << "Please input source file." << endl;
		abort();
	}
}

//----------------------------------------------------------------------------
// CPU Jpeg圧縮テストルーチン
//============================================================================
void cpu_exec(int argc, char *argv[]) {
	StopWatch watch(StopWatch::CPU_OPTIMUM);

	// コマンドライン引数からファイル名取得
	string file_name, out_file_name;
	parse_arg(argc, argv, file_name, out_file_name);

	// 画像データを読み出し
	BitmapCVUtil source(file_name, BitmapCVUtil::RGB_COLOR);
	const int width = source.getWidth();
	const int height = source.getHeight();

	std::cout << "===============================================" << std::endl;
	std::cout << " Start CPU Encoding & Decoding" << std::endl;
	std::cout << "-----------------------------------------------\n" << std::endl;

	ByteBuffer encode_result;
	{
		std::cout << "-----------------------------------------------" << std::endl;
		std::cout << " Encode" << std::endl;
		std::cout << "-----------------------------------------------" << std::endl;

		JpegEncoder encoder(width, height);

		watch.start();
		size_t data_size = encoder.encode((byte*) source.getRawData(), width * height * 3,
			encode_result);
		watch.lap();
		watch.stop();
		std::cout << watch.getLastElapsedTime() << "[ms]\n" << std::endl;
	}

	BitmapCVUtil result(width, height, 8, source.getBytePerPixel());
	{
		std::cout << "-----------------------------------------------" << std::endl;
		std::cout << " Decode" << std::endl;
		std::cout << "-----------------------------------------------" << std::endl;

		JpegDecoder decoder(width, height);

		watch.start();
		decoder.decode(encode_result.data(), encode_result.size(), (byte*) result.getRawData(),
			width * height * 3);
		watch.lap();
		watch.stop();
		std::cout << watch.getLastElapsedTime() << "[ms]\n" << std::endl;
	}

	result.saveToFile("cpu_decoder_" + out_file_name);

	std::cout << "-----------------------------------------------" << std::endl;
	std::cout << " Finish CPU Encoding & Decoding" << std::endl;
	std::cout << "===============================================\n\n" << std::endl;
}
