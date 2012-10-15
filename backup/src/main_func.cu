#include <cstdlib>
#include <cstring>
#include <cmath>

#ifdef __linux__
#include <sys/time.h>
#include <sys/resource.h>
#else
int a;
#endif

#include <cuda_runtime.h>
#include <iostream>
using namespace std;

#include "cpu_jpeg.h"
#include "gpu_jpeg.cuh"

#include "utils/timer.h"
#include "utils/util_cv.h"
#include "utils/encoder_tables.h"
#include "utils/gpu_in_bit_stream.cuh"

const double kDisSqrt2 = 1.0 / 1.41421356; // 2の平方根の逆数
const double kPaiDiv16 = 3.14159265 / 16; // 円周率/16

void parse_arg(int argc, char *argv[], string &in_file, string &out_file) {
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
	for (int i = 0; i < watch.getLapCount(); ++i) {
		cout << watch.getLapList()[i] << ",";
	}
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

	cout << "--start color conversionT." << endl;
	watch.start();
	color_trans_yuv_to_rgb(c_img_ycc, (unsigned char*) (p_cvimg_dst->im.p_buf),
		sizeX, sizeY);
	watch.lap();
	watch.stop();

	cout << "@end timer." << endl;
	cout << "\n\nCPU DECODING STATE\n" << "time:" << watch.getLastElapsedTime()
		<< "[sec]\n\n" << endl;

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

void set_dct_coefficient(float CosT[8][8], float ICosT[8][8]) {
//cos,chenのアルゴリズムを使う場合不要
	for (int y = 0; y < 8; y++) {
		for (int x = 0; x < 8; x++) {
			CosT[y][x] = (y == 0 ? kDisSqrt2 : 1)
				* cos((2 * x + 1) * y * kPaiDiv16);
		}
	}
//IDCTの方はCu,Cvを別で考えなくちゃいけない
	for (int y = 0; y < 8; y++) {
		for (int x = 0; x < 8; x++) {
			ICosT[y][x] = cos((2 * x + 1) * y * kPaiDiv16);
		}
	}
}

void gpu_exec(int argc, char *argv[]) {
	StopWatch watch(StopWatch::CPU_OPTIMUM);

//----------------------------------------------------------------------------
// 画像読み込み
//============================================================================
	cout << "start gpu process." << endl;
	cout << "@start timer." << endl;
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	watch.start();
	cudaEventRecord(start, 0);

	cout << "--load image." << endl;
	string file_name, out_file_name;
	parse_arg(argc, argv, file_name, out_file_name);

	UtilCVImage *p_cvimg_src = utilCV_LoadImage(file_name.c_str(),
		UTIL_CVIM_COLOR);
	const int sizeX = p_cvimg_src->im.width, sizeY = p_cvimg_src->im.height;
	const int size = sizeX * sizeY;
	const int C_size = sizeX * sizeY / 4;
	const int YCC_size = size + sizeX * sizeY / 2;

//----------------------------------------------------------------------------
// 色変換テーブルの作成
//============================================================================
	cout << "--start create conversion table." << endl;
	int *trans_table_Y = (int*) malloc(sizeof(int) * sizeX * sizeY);
	int *trans_table_C = (int*) malloc(sizeof(int) * sizeX * sizeY);
	make_trans_table(trans_table_Y, trans_table_C, sizeX, sizeY);

	int *itrans_table_Y = (int*) malloc(sizeof(int) * sizeX * sizeY);
	int *itrans_table_C = (int*) malloc(sizeof(int) * sizeX * sizeY);
	make_itrans_table(itrans_table_Y, itrans_table_C, sizeX, sizeY);

//なぜかこれより前でcudaMallocするとsegmentation fault起きる(らしい)

//----------------------------------------------------------------------------
// DCT係数のセット
//============================================================================
	cout << "--start set_dct_coefficient." << endl;
	float CosT[8][8];
	float ICosT[8][8];
	set_dct_coefficient(CosT, ICosT);

//----------------------------------------------------------------------------
// ハフマン符号化用メモリ確保
//============================================================================
	cout << "--start allocate device memory(for encode)." << endl;
	GPUOutBitStream *GPUmOBSP = new GPUOutBitStream[YCC_size / 64];
	GPUOutBitStream *GPUmOBSP_d;
	cudaMalloc((void**) &GPUmOBSP_d, sizeof(GPUOutBitStream) * (YCC_size / 64));
	cudaMemcpy(GPUmOBSP_d, GPUmOBSP, sizeof(GPUOutBitStream) * (YCC_size / 64),
		cudaMemcpyHostToDevice);
	byte* dst_NumBits = new byte[(sizeX * sizeY + 2 * (sizeX / 2) * (sizeY / 2))
		/ 64];

	GPUOutBitStreamBufferPointer buf_d;
	cudaMalloc((void**) &(buf_d.HeadOfBuf),
		sizeof(byte) * (YCC_size / 64) * MBS);
	InitGPUBuffer(&buf_d, (YCC_size / 64) * MBS);

	byte *mHeadOfBufP_d;
	cudaMalloc((void**) &mHeadOfBufP_d, sizeof(byte) * (size * 3)); //*3
	cudaMemset(mHeadOfBufP_d, 0, sizeof(byte) * (size * 3));

//----------------------------------------------------------------------------
// Encode用定数転送,コンスタントメモリも使ってみたい
//============================================================================
	cout << "--start allocate device memory(constant for encode)." << endl;
//先に送っておくもの
	int *trans_Y_d, *trans_C_d;
	cudaMalloc((void**) &trans_Y_d, sizeof(int) * sizeX * sizeY);
	cudaMemcpy(trans_Y_d, trans_table_Y, sizeof(int) * sizeX * sizeY,
		cudaMemcpyHostToDevice);
	cudaMalloc((void**) &trans_C_d, sizeof(int) * sizeX * sizeY);
	cudaMemcpy(trans_C_d, trans_table_C, sizeof(int) * sizeX * sizeY,
		cudaMemcpyHostToDevice);

	float *CosT_d;
	cudaMalloc((void**) &CosT_d, sizeof(float) * 64);
	cudaMemcpy(CosT_d, CosT, sizeof(float) * 64, cudaMemcpyHostToDevice);

	int *Qua_Y_d, *Qua_C_d;
	cudaMalloc((void**) &Qua_Y_d, sizeof(int) * 64);
	cudaMemcpy(Qua_Y_d, kYQuantumT, sizeof(int) * 64, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &Qua_C_d, sizeof(int) * 64);
	cudaMemcpy(Qua_C_d, kCQuantumT, sizeof(int) * 64, cudaMemcpyHostToDevice);

	int *zigzag_d;
	cudaMalloc((void**) &zigzag_d, sizeof(int) * 64);
	cudaMemcpy(zigzag_d, kZigzagT, sizeof(int) * 64, cudaMemcpyHostToDevice);

//----------------------------------------------------------------------------
// Decode用定数転送,コンスタントメモリも使ってみたい
//============================================================================
	cout << "--start allocate device memory(constant for decode)." << endl;
	int *itrans_Y_d, *itrans_C_d;
	cudaMalloc((void**) &itrans_Y_d, sizeof(int) * sizeX * sizeY);
	cudaMemcpy(itrans_Y_d, itrans_table_Y, sizeof(int) * sizeX * sizeY,
		cudaMemcpyHostToDevice);
	cudaMalloc((void**) &itrans_C_d, sizeof(int) * sizeX * sizeY);
	cudaMemcpy(itrans_C_d, itrans_table_C, sizeof(int) * sizeX * sizeY,
		cudaMemcpyHostToDevice);

	float *ICosT_d;
	cudaMalloc((void**) &ICosT_d, sizeof(float) * 64);
	cudaMemcpy(ICosT_d, ICosT, sizeof(float) * 64, cudaMemcpyHostToDevice);

	int *ycc_d;
	cudaMalloc((void**) &ycc_d,
		sizeof(int) * (sizeX * sizeY + 2 * (sizeX / 2) * (sizeY / 2)));
	cudaMemset(ycc_d, 0,
		sizeof(int) * (sizeX * sizeY + 2 * (sizeX / 2) * (sizeY / 2)));

//----------------------------------------------------------------------------
// Decode用メモリ確保,コンスタントメモリも使ってみたい
//============================================================================
	cout << "--start allocate device memory(for decode)." << endl;
	int *coef_d;
	cudaMalloc((void**) &coef_d,
		sizeof(int) * (sizeX * sizeY + 2 * (sizeX / 2) * (sizeY / 2)));
	cudaMemset(coef_d, 0,
		sizeof(int) * (sizeX * sizeY + 2 * (sizeX / 2) * (sizeY / 2)));

	float *f_d;
	cudaMalloc((void**) &f_d,
		sizeof(float) * (sizeX * sizeY + 2 * (sizeX / 2) * (sizeY / 2)));
	cudaMemset(f_d, 0,
		sizeof(float) * (sizeX * sizeY + 2 * (sizeX / 2) * (sizeY / 2)));

	int *qua_d;
	cudaMalloc((void**) &qua_d,
		sizeof(int) * (sizeX * sizeY + 2 * (sizeX / 2) * (sizeY / 2)));
	cudaMemset(qua_d, 0,
		sizeof(int) * (sizeX * sizeY + 2 * (sizeX / 2) * (sizeY / 2)));

	byte *src_d;
	cudaMalloc((void**) &src_d, sizeof(byte) * sizeX * sizeY * 3);

//----------------------------------------------------------------------------
// カーネルDimension設定
//============================================================================
	cout << "end preprocess." << endl;
	const int THREADS = 256;

	const int DCT4_TH = 1;

	const int QUA0_TH = 64;
	const int QUA1_TH = 64;

	const int HUF0_TH = 16;
	const int HUF1_TH = 4; //divide使うなら最速

	dim3 Dg0_0(sizeX * sizeY / THREADS, 1, 1), Db0_0(THREADS, 1, 1);
	dim3 Dg0_1(sizeX * sizeY / THREADS / 2, 1, 1), Db0_1(sizeY / 2, 1, 1);

	dim3 Dg1((sizeX * sizeY + 2 * (sizeX / 2) * (sizeY / 2)) / 64 / DCT4_TH, 1,
		1), Db1(DCT4_TH, 8, 8); //DCT4_THは16が最大

	dim3 Dg2_0(size / QUA0_TH, 1, 1), Db2_0(QUA0_TH, 1, 1);
	dim3 Dg2_1((2 * C_size) / QUA1_TH, 1, 1), Db2_1(QUA1_TH, 1, 1);

	dim3 Dg3_0(YCC_size / 64 / HUF0_TH, 1, 1), Db3_0(HUF0_TH, 1, 1); //YCC_size
	dim3 Dg3_1(YCC_size / 64 / HUF1_TH, 1, 1), Db3_1(HUF1_TH, 1, 1); //YCC_size

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	float etime;
	cudaEventElapsedTime(&etime, start, end);
	etime /= 1000.0;
	watch.stop();
	cout << "@end timer." << endl;
	cout << "\n\nPRE-PROCESS STATE\n" << "time:" << etime << "[sec]\n"
		<< "time:" << watch.getLastElapsedTime() << "[sec]*cpu\n\n" << endl;

//----------------------------------------------------------------------------
//
//
//
// ここより前は前処理想定
//
//
//
//
//============================================================================

//----------------------------------------------------------------------------
//
//
// Encode
//
//
//
//
//
//============================================================================

// ループ想定
//----------------------------------------------------------------------------
// 画像読み込み
//============================================================================
	cout << "--load image." << endl;
	UtilCVImage *p_cvimg_dst = utilCV_CreateImage(sizeX, sizeY, 8,
		p_cvimg_src->im.px_byte);

//----------------------------------------------------------------------------
// メモリ転送
//============================================================================
	cout << "--start gpu encode." << endl;
	cout << "@start timer." << endl;
	watch.start();
	cudaEventRecord(start, 0);

	cout << "--transfer memory to device." << endl;
	cudaMemcpy(src_d, (unsigned char*) p_cvimg_src->im.p_buf,
		sizeof(byte) * sizeX * sizeY * 3, cudaMemcpyHostToDevice);

//----------------------------------------------------------------------------
// RGB->yuv
//============================================================================
	cout << "--start color conversion." << endl;
	gpu_color_trans_Y<<<Dg0_0, Db0_0>>>(src_d, ycc_d, trans_Y_d);
	gpu_color_trans_C<<<Dg0_0, Db0_0>>>(src_d, ycc_d, trans_C_d, sizeY, C_size);

//----------------------------------------------------------------------------
// DCT
//============================================================================
	cout << "--start DCT." << endl;
	gpu_dct_0<<<Dg1, Db1>>>(ycc_d, f_d, CosT_d);
	gpu_dct_1<<<Dg1, Db1>>>(f_d, coef_d, CosT_d);

//----------------------------------------------------------------------------
// 量子化
//============================================================================
	cout << "--start gpu_zig_quantize." << endl;
	gpu_zig_quantize_Y<<<Dg2_0, Db2_0>>>(coef_d, qua_d, zigzag_d, Qua_Y_d);
	gpu_zig_quantize_C<<<Dg2_1, Db2_1>>>(coef_d, qua_d, zigzag_d, Qua_C_d,
		size);

//----------------------------------------------------------------------------
// ハフマン符号化
//============================================================================
	cout << "--start huffman_encode." << endl;
	gpu_huffman_mcu<<<Dg3_0, Db3_0>>>(qua_d, GPUmOBSP_d, buf_d.WriteBufAddress,
		buf_d.EndOfBuf, sizeX, sizeY);

// 逐次処理のためCPUに戻す
	cudaMemcpy(GPUmOBSP, GPUmOBSP_d, sizeof(GPUOutBitStream) * (YCC_size / 64),
		cudaMemcpyDeviceToHost);
	cpu_huffman_middle(GPUmOBSP, sizeX, sizeY, dst_NumBits);
	cudaMemcpy(GPUmOBSP_d, GPUmOBSP, sizeof(GPUOutBitStream) * (YCC_size / 64),
		cudaMemcpyHostToDevice);

	gpu_huffman_write_devide0<<<Dg3_1, Db3_1>>>(GPUmOBSP_d,
		buf_d.WriteBufAddress, mHeadOfBufP_d, sizeX, sizeY);
	gpu_huffman_write_devide1<<<Dg3_1, Db3_1>>>(GPUmOBSP_d,
		buf_d.WriteBufAddress, mHeadOfBufP_d, sizeX, sizeY);
	gpu_huffman_write_devide2<<<Dg3_1, Db3_1>>>(GPUmOBSP_d,
		buf_d.WriteBufAddress, mHeadOfBufP_d, sizeX, sizeY);

//----------------------------------------------------------------------------
// 結果メモリ転送 :出力は「dst_dataとdst_NumBits」の２つ
//============================================================================
	cout << "--transfer memory to host." << endl;
	int dst_size = GPUmOBSP[YCC_size / 64 - 1].mBytePos
		+ (GPUmOBSP[YCC_size / 64 - 1].mBitPos == 7 ? 0 : 1);
	char *dst_data = (char *) malloc(sizeof(char) * dst_size);
	cudaMemcpy(dst_data, mHeadOfBufP_d, sizeof(byte) * dst_size,
		cudaMemcpyDeviceToHost);

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&etime, start, end);
	etime /= 1000.0;
	watch.stop();
	cout << "@end timer." << endl;
	cout << "\n\nGPU ENCODING STATE\n" << "size:" << sizeX * sizeY * 3 << " -> "
		<< dst_size << "\n" << "time:" << etime << "[sec]\n" << "time:"
		<< watch.getLastElapsedTime() << "[sec]*cpu\n\n" << endl;

//----------------------------------------------------------------------------
//
//
// Decode
//
//
//
//
//
//============================================================================
	cout << "--start gpu decode." << endl;
	cout << "@start timer." << endl;
	cudaEventRecord(start, 0);
	watch.start();

//----------------------------------------------------------------------------
// メモリ確保
//============================================================================
	cout << "--start memory allocation." << endl;
	InBitStream *mIBSP = new InBitStream(dst_data, dst_size);
	int* c_qua = (int*) (malloc(
		sizeof(int) * (sizeX * sizeY + 2 * (sizeX / 2) * (sizeY / 2))));

//並列展開するためのサイズ情報を入れるための構造体
	GPUInBitStream *GPUmIBSP = new GPUInBitStream[YCC_size / 64]; //
	GPUInBitStream *GPUmIBSP_d;
	cudaMalloc((void**) &GPUmIBSP_d, sizeof(GPUInBitStream) * (YCC_size / 64));
	cudaMemcpy(GPUmIBSP_d, GPUmIBSP, sizeof(GPUInBitStream) * (YCC_size / 64),
		cudaMemcpyHostToDevice);

//生データを入れるためのバッファ（全体バッファ）
	GPUCInBitStream_BufP IbufP;
	cudaMalloc((void**) &(IbufP.mHeadOfBufP), dst_size);
	cudaMemset(IbufP.mHeadOfBufP, 0, dst_size);

//----------------------------------------------------------------------------
// ハフマン復号
//============================================================================
	cout << "--start decode_huffman." << endl;
// CPU
	decode_huffman(mIBSP, c_qua, sizeX, sizeY);
// GPU:GPUInstream.hにバグがある可能性もあるので留意
	cudaMemcpy(qua_d, c_qua,
		sizeof(int) * (sizeX * sizeY + 2 * (sizeX / 2) * (sizeY / 2)),
		cudaMemcpyHostToDevice);

//----------------------------------------------------------------------------
// 逆量子化
//============================================================================
	cout << "--start gpu_izig_quantize." << endl;
	gpu_izig_quantize_Y<<<Dg2_0, Db2_0>>>(qua_d, coef_d, zigzag_d, Qua_Y_d);
	gpu_izig_quantize_C<<<Dg2_1, Db2_1>>>(qua_d, coef_d, zigzag_d, Qua_C_d,
		size);

//----------------------------------------------------------------------------
// 逆DCT
//============================================================================
	cout << "--start Inv-DCT." << endl;
	gpu_idct_0<<<Dg1, Db1>>>(coef_d, f_d, ICosT_d);
	gpu_idct_1<<<Dg1, Db1>>>(f_d, ycc_d, ICosT_d);

//----------------------------------------------------------------------------
// yuv->RGB
//============================================================================
	cout << "--start color conversion." << endl;
	gpu_color_itrans<<<Dg0_0, Db0_0>>>(ycc_d, src_d, itrans_Y_d, itrans_C_d,
		C_size);

//----------------------------------------------------------------------------
// 結果転送
//============================================================================
	cudaMemcpy((byte*) p_cvimg_dst->im.p_buf, src_d,
		sizeof(byte) * sizeX * sizeY * 3, cudaMemcpyDeviceToHost);
	watch.stop();
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&etime, start, end);
	etime /= 1000.0;
	cout << "@end timer." << endl;
	cout << "\n\nGPU DECODING STATE\n" << "time:" << etime << "[sec]\n"
		<< "time:" << watch.getLastElapsedTime() << "[sec]*cpu\n\n" << endl;

	cout << "save imege..." << endl;
	out_file_name = "gpu_" + out_file_name;
	utilCV_SaveImage(out_file_name.c_str(), p_cvimg_dst);

	cout << "end gpu process." << endl;
	cout << "------------------------------------------------------\n\n"
		<< endl;

//----------------------------------------------------------------------------
// 開放処理
//============================================================================
	free(trans_table_Y);
	free(trans_table_C);

	free(c_qua);
	free(dst_data);

	delete[] GPUmOBSP;

	cudaFree(trans_Y_d);
	cudaFree(trans_C_d);
	cudaFree(CosT_d);
	cudaFree(Qua_Y_d);
	cudaFree(Qua_C_d);
	cudaFree(zigzag_d);

	cudaFree(itrans_Y_d);
	cudaFree(itrans_C_d);
	cudaFree(ICosT_d);

	cudaFree(src_d);
	cudaFree(ycc_d);
	cudaFree(coef_d);
	cudaFree(f_d);
	cudaFree(qua_d);
	cudaFree(GPUmOBSP_d);
	cudaFree(buf_d.HeadOfBuf);
	cudaFree(mHeadOfBufP_d);

	utilCV_ReleaseImage(&p_cvimg_src);
	utilCV_ReleaseImage(&p_cvimg_dst);

	delete mIBSP;
}

