#include <cstdlib>
#include <cstring>

#include <fstream>
#include <iostream>

using namespace std;

#include "cpu_jpeg.h"

#include "utils/timer.h"
#include "utils/util_cv.h"
#include "utils/encoder_tables.h"

const double kDisSqrt2 = 1.0 / 1.41421356; // 2の平方根の逆数
const double kPaiDiv16 = 3.14159265 / 16; // 円周率/16

static void parse_arg(int argc, char *argv[], string &in_file, string &out_file) {
	if (argc == 3) {
		in_file = argv[1];
		out_file = argv[2];
	} else {
		cout << "Please input source file." << endl;
		abort();
	}
}

void cpu_exec(int argc, char *argv[]) {
	StopWatch watch(StopWatch::CPU_OPTIMUM);

	string file_name, out_file_name;
	parse_arg(argc, argv, file_name, out_file_name);

//----------------------------------------------------------------------------
// Encode
//============================================================================
	cout << "start cpu encoding." << endl;

	cout << "--load image." << endl;

	UtilCVImage* p_cvimg_src = utilCV_LoadImage(file_name.c_str(),
		UTIL_CVIM_COLOR);
	const int sizeX = p_cvimg_src->im.width, sizeY = p_cvimg_src->im.height;

	OutBitStream OBSP(sizeX * sizeY * 3);
	UtilCVImage* p_cvimg_dst = utilCV_CreateImage(sizeX, sizeY, 8,
		p_cvimg_src->im.px_byte);
	int* c_img_ycc = (int*) (malloc(
		sizeof(int) * (sizeX * sizeY + 2 * (sizeX / 2) * (sizeY / 2))));
	int* c_coef = (int*) (malloc(
		sizeof(int) * (sizeX * sizeY + 2 * (sizeX / 2) * (sizeY / 2))));
	int* c_qua = (int*) (malloc(
		sizeof(int) * (sizeX * sizeY + 2 * (sizeX / 2) * (sizeY / 2))));

	cout << "@start timer." << endl;
	cout << "--start color conversion." << endl;
	watch.start();
	color_trans_rgb_to_yuv((unsigned char*) (p_cvimg_src->im.p_buf), c_img_ycc,
		sizeX, sizeY);
	watch.lap();
	watch.stop();

	cout << "--start DCT." << endl;
	watch.start();
	dct(c_img_ycc, c_coef, sizeX, sizeY);
	watch.lap();
	watch.stop();

	cout << "--start zig_quantize." << endl;
	watch.start();
	zig_quantize(c_coef, c_qua, sizeX, sizeY);
	watch.lap();
	watch.stop();

	cout << "--start huffman_encode." << endl;
	watch.start();
	huffman_encode(c_qua, &OBSP, sizeX, sizeY);
	char* dst_data;
	int dst_size = OBSP.GetStreamSize();
	dst_data = (char*) (malloc(dst_size));
	memcpy(dst_data, OBSP.GetStreamAddress(), dst_size);
	watch.lap();
	watch.stop();

	cout << "@end timer." << endl;
	cout << "\n\nCPU ENCODING STATE\n" << "size:" << sizeX * sizeY * 3 << " -> "
		<< dst_size << "\n" << "time:" << watch.getTotalTime() << "[sec]"
		<< endl;

	ofstream ofs("result.csv", std::ios::app);
	ofs << "cpu encode:" << file_name << endl;
	ofs << "color conversion," << "DCT," << "zig_quantize," << "huffman,"
		<< "total" << endl;
	for (int i = 0; i < watch.getLapCount(); ++i) {
		ofs << watch.getLapList()[i] << ",";
	}
	ofs << watch.getTotalTime() << "\n" << endl;
	watch.clear();

	cout << "\n\n" << endl;

//----------------------------------------------------------------------------
// Decode
//============================================================================
	cout << "start cpu decoding." << endl;
	cout << "@start timer." << endl;
	watch.start();
	InBitStream IBSP(dst_data, dst_size);
	watch.stop();
	watch.lap();
	watch.stop();

	cout << "--start decode_huffman." << endl;
	watch.start();
	decode_huffman(&IBSP, c_qua, sizeX, sizeY);
	watch.lap();
	watch.stop();

	cout << "--start izig_quantize." << endl;
	watch.start();
	izig_quantize(c_qua, c_coef, sizeX, sizeY);
	watch.lap();
	watch.stop();

	cout << "--start Inv-DCT." << endl;
	watch.start();
	idct(c_coef, c_img_ycc, sizeX, sizeY);
	watch.lap();
	watch.stop();

	cout << "--start color conversion." << endl;
	watch.start();
	color_trans_yuv_to_rgb(c_img_ycc, (unsigned char*) (p_cvimg_dst->im.p_buf),
		sizeX, sizeY);
	watch.lap();
	watch.stop();

	cout << "@end timer." << endl;
	cout << "\n\nCPU DECODING STATE\n" << "time:" << watch.getTotalTime()
		<< "[sec]\n\n" << endl;
	cout << "allocation," << "huffman," << "zig_quantize," << "Inv-DCT,"
		<< "color conversion " << endl;
	for (int i = 0; i < watch.getLapCount(); ++i) {
		cout << watch.getLapList()[i] << ",";
	}

	ofs << "cpu decode:" << file_name << endl;
	ofs << "allocation," << "huffman," << "zig_quantize," << "Inv-DCT,"
		<< "color conversion," << "total" << endl;
	for (int i = 0; i < watch.getLapCount(); ++i) {
		ofs << watch.getLapList()[i] << ",";
	}
	ofs << watch.getTotalTime() << "\n" << endl;

	cout << "save image..." << endl;
	out_file_name = "cpu_" + out_file_name;
	utilCV_SaveImage(out_file_name.c_str(), p_cvimg_dst);

	free(c_img_ycc);
	free(c_coef);
	free(c_qua);
	free(dst_data);
	utilCV_ReleaseImage(&p_cvimg_src);
	utilCV_ReleaseImage(&p_cvimg_dst);

	cout << "end cpu process." << endl;
	cout << "------------------------------------------------------\n\n"
		<< endl;
}
