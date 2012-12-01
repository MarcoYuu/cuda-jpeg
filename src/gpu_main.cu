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

	std::cout << "===============================================" << std::endl;
	std::cout << " Start GPU Encoding & Decoding" << std::endl;
	std::cout << "-----------------------------------------------\n" << std::endl;

	std::cout << "	-----------------------------------------------" << std::endl;
	std::cout << "	 Encode" << std::endl;
	std::cout << "	-----------------------------------------------" << std::endl;
	int result_size;
	cuda_memory<byte> encode_result(sizeof(byte) * (width * height * 3));
	encode_result.fill_zero();
	{
		jpeg::ohmura::JpegEncoder encoder(width, height);

		watch.start();
		{
			result_size = encoder.encode((byte*) source.getRawData(), encode_result);
			encode_result.sync_to_host();
		}
		watch.stop();
		std::cout << "	" << watch.getLastElapsedTime() * 1000 << "[ms]\n" << std::endl;
	}

	watch.clear();
	std::cout << "	-----------------------------------------------" << std::endl;
	std::cout << "	 Decode" << std::endl;
	std::cout << "	-----------------------------------------------" << std::endl;
	BitmapCVUtil result(width, height, 8, source.getBytePerPixel());
	{
		device_memory<byte> decode_result(width * height * 3);
		jpeg::ohmura::JpegDecoder decoder(width, height);

		watch.start();
		{
			decoder.decode(encode_result.host_data(), result_size, decode_result);
			decode_result.copy_to_host((byte*) result.getRawData(), decode_result.size());
		}
		watch.stop();
		std::cout << "	" << watch.getLastElapsedTime() * 1000 << "[ms]\n" << std::endl;
	}
	result.saveToFile("gpu_" + out_file_name);

	std::cout << "-----------------------------------------------" << std::endl;
	std::cout << " Finish GPU Encoding & Decoding" << std::endl;
	std::cout << "===============================================\n\n" << std::endl;
}

