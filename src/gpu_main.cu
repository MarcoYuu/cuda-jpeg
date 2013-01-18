#include <cstdlib>
#include <cstring>
#include <cmath>

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

#include "jpeg/cuda/ohmura/gpu_jpeg.cuh"

#include "utils/util_cv.h"
#include "utils/timer.h"
#include "utils/cuda/cuda_timer.h"
#include "utils/cuda/cuda_memory.hpp"

void gpu_main(const std::string &file_name, const std::string &out_file_name) {
	using namespace std;
	using namespace util;
	using namespace util::cuda;
	using namespace jpeg;
	using namespace jpeg::ohmura;

	CudaStopWatch watch;

	//----------------------------------------------------------------------------
	// 画像読み込み
	//============================================================================
	BitmapCVUtil source(file_name, BitmapCVUtil::RGB_COLOR);
	const int width = source.getWidth();
	const int height = source.getHeight();

	std::cout << "Encode" << std::endl;
	int result_size;

	watch.start();
	cuda_memory<byte> encode_result(sizeof(byte) * (width * height * 3));
	encode_result.fill_zero();
	watch.stop();
	cout << "Preprocess, " << watch.getLastElapsedTime() << endl;
	{
		watch.start();
		jpeg::ohmura::JpegEncoder encoder(width, height);
		watch.stop();
		cout << "Preprocess, " << watch.getLastElapsedTime() << endl;

		{
			result_size = encoder.encode((byte*) source.getRawData(), encode_result);

			watch.start();
			encode_result.sync_to_host();
			watch.stop();
			cout << "Memory Transfer, " << watch.getLastElapsedTime() << endl;
		}
	}

	watch.clear();

	std::cout << "\nDecode" << std::endl;
	watch.start();
	BitmapCVUtil result(width, height, 8, source.getBytePerPixel());
	watch.stop();
	cout << "Preprocess, " << watch.getLastElapsedTime() << endl;

	{
		watch.start();
		device_memory<byte> decode_result(width * height * 3);
		jpeg::ohmura::JpegDecoder decoder(width, height);
		watch.stop();
		cout << "Preprocess, " << watch.getLastElapsedTime() << endl;

		{
			decoder.decode(encode_result.host_data(), result_size, decode_result);

			watch.start();
			decode_result.copy_to_host((byte*) result.getRawData(), decode_result.size());
			watch.stop();
			cout << "Memory Transfer, " << watch.getLastElapsedTime() << "\n" << endl;
		}
	}
	result.saveToFile("gpu_" + out_file_name);
}

