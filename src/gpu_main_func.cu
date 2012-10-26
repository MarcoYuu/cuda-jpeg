#include <cstdlib>
#include <cstring>
#include <cmath>

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

using namespace std;

#include "cpu_jpeg.h"
#include "gpu_jpeg.cuh"

#include "utils/util_cv.h"
#include "utils/timer.h"
#include "utils/cuda_timer.h"
#include "utils/cuda_memory.hpp"
#include "utils/util_cv.h"
#include "utils/encoder_tables.h"
#include "utils/in_bit_stream.h"
#include "utils/gpu_in_bit_stream.cuh"

const double kDisSqrt2 = 1.0 / 1.41421356; // 2の平方根の逆数
const double kPaiDiv16 = 3.14159265 / 16; // 円周率/16

static void parse_arg(int argc, char *argv[], string &in_file,
	string &out_file) {
	if (argc == 3) {
		in_file = argv[1];
		out_file = argv[2];
	} else {
		cout << "Please input source file." << endl;
		abort();
	}
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
	//----------------------------------------------------------------------------
	// 画像読み込み
	//============================================================================
	string file_name, out_file_name;
	parse_arg(argc, argv, file_name, out_file_name);

	BitmapCVUtil source(file_name, BitmapCVUtil::RGB_COLOR);
	const int width = source.getWidth();
	const int height = source.getHeight();
	const int Y_size = width * height;
	const int C_size = Y_size / 4;
	const int YCC_size = Y_size + C_size * 2;

	//----------------------------------------------------------------------------
	// 色変換テーブルの作成
	//============================================================================
	CudaMemory<int> trans_table_Y(width * height);
	CudaMemory<int> trans_table_C(width * height);
	make_trans_table(trans_table_Y.host_data(), trans_table_C.host_data(), width, height);

	CudaMemory<int> itrans_table_Y(width * height);
	CudaMemory<int> itrans_table_C(width * height);
	make_itrans_table(itrans_table_Y.host_data(), itrans_table_C.host_data(), width, height);

	//----------------------------------------------------------------------------
	// DCT係数のセット
	//============================================================================
	float CosT[8][8];
	float ICosT[8][8];
	set_dct_coefficient(CosT, ICosT);

	//----------------------------------------------------------------------------
	// ハフマン符号化用メモリ確保
	//============================================================================
	CudaMemory<GPUOutBitStream> GPUmOBSP(YCC_size / 64);
	GPUmOBSP.syncDeviceMemory();

	ByteBuffer dst_NumBits((width * height + 2 * (width / 2) * (height / 2)) / 64);

	GPUOutBitStreamBufferPointer buf_d(sizeof(byte) * (YCC_size / 64) * MBS);

	byte *mHeadOfBufP_d;
	cudaMalloc((void**) &mHeadOfBufP_d, sizeof(byte) * (Y_size * 3)); //*3
	cudaMemset(mHeadOfBufP_d, 0, sizeof(byte) * (Y_size * 3));

	//----------------------------------------------------------------------------
	// Encode用定数転送,コンスタントメモリも使ってみたい
	//============================================================================
	//先に送っておくもの
	trans_table_C.syncDeviceMemory();
	trans_table_Y.syncDeviceMemory();

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
	itrans_table_C.syncDeviceMemory();
	itrans_table_Y.syncDeviceMemory();

	float *ICosT_d;
	cudaMalloc((void**) &ICosT_d, sizeof(float) * 64);
	cudaMemcpy(ICosT_d, ICosT, sizeof(float) * 64, cudaMemcpyHostToDevice);

	int *ycc_d;
	cudaMalloc((void**) &ycc_d, sizeof(int) * (width * height + 2 * (width / 2) * (height / 2)));
	cudaMemset(ycc_d, 0, sizeof(int) * (width * height + 2 * (width / 2) * (height / 2)));

	//----------------------------------------------------------------------------
	// Decode用メモリ確保,コンスタントメモリも使ってみたい
	//============================================================================
	int *coef_d;
	cudaMalloc((void**) &coef_d, sizeof(int) * (width * height + 2 * (width / 2) * (height / 2)));
	cudaMemset(coef_d, 0, sizeof(int) * (width * height + 2 * (width / 2) * (height / 2)));

	float *f_d;
	cudaMalloc((void**) &f_d, sizeof(float) * (width * height + 2 * (width / 2) * (height / 2)));
	cudaMemset(f_d, 0, sizeof(float) * (width * height + 2 * (width / 2) * (height / 2)));

	int *qua_d;
	cudaMalloc((void**) &qua_d, sizeof(int) * (width * height + 2 * (width / 2) * (height / 2)));
	cudaMemset(qua_d, 0, sizeof(int) * (width * height + 2 * (width / 2) * (height / 2)));

	byte *src_d;
	cudaMalloc((void**) &src_d, sizeof(byte) * width * height * 3);

	//----------------------------------------------------------------------------
	// カーネルDimension設定
	//============================================================================
	const int THREADS = 256;

	const int DCT4_TH = 1;

	const int QUA0_TH = 64;
	const int QUA1_TH = 64;

	const int HUF0_TH = 16;
	const int HUF1_TH = 4; //divide使うなら最速

	dim3 Dg0_0(width * height / THREADS, 1, 1), Db0_0(THREADS, 1, 1);
	dim3 Dg0_1(width * height / THREADS / 2, 1, 1), Db0_1(height / 2, 1, 1);

	dim3 Dg1((width * height + 2 * (width / 2) * (height / 2)) / 64 / DCT4_TH, 1, 1), Db1(DCT4_TH, 8, 8); //DCT4_THは16が最大

	dim3 Dg2_0(Y_size / QUA0_TH, 1, 1), Db2_0(QUA0_TH, 1, 1);
	dim3 Dg2_1((2 * C_size) / QUA1_TH, 1, 1), Db2_1(QUA1_TH, 1, 1);

	dim3 Dg3_0(YCC_size / 64 / HUF0_TH, 1, 1), Db3_0(HUF0_TH, 1, 1); //YCC_size
	dim3 Dg3_1(YCC_size / 64 / HUF1_TH, 1, 1), Db3_1(HUF1_TH, 1, 1); //YCC_size

	//----------------------------------------------------------------------------
	// ここより前は前処理想定
	//============================================================================

	//----------------------------------------------------------------------------
	// Encode
	//============================================================================
	//----------------------------------------------------------------------------
	// 画像読み込み
	//============================================================================
	BitmapCVUtil result(width, height, 8, source.getBytePerPixel());

	//----------------------------------------------------------------------------
	// メモリ転送
	//============================================================================
	cudaMemcpy(src_d, source.getRawData(), sizeof(byte) * width * height * 3, cudaMemcpyHostToDevice);

	//----------------------------------------------------------------------------
	// RGB->yuv
	//============================================================================
	gpu_color_trans_Y<<<Dg0_0, Db0_0>>>(src_d, ycc_d, trans_table_Y.device_data());
	gpu_color_trans_C<<<Dg0_0, Db0_0>>>(src_d, ycc_d, trans_table_C.device_data(), height, C_size);

	//----------------------------------------------------------------------------
	// DCT
	//============================================================================
	gpu_dct_0<<<Dg1, Db1>>>(ycc_d, f_d, CosT_d);
	gpu_dct_1<<<Dg1, Db1>>>(f_d, coef_d, CosT_d);

	//----------------------------------------------------------------------------
	// 量子化
	//============================================================================
	gpu_zig_quantize_Y<<<Dg2_0, Db2_0>>>(coef_d, qua_d, zigzag_d, Qua_Y_d);
	gpu_zig_quantize_C<<<Dg2_1, Db2_1>>>(coef_d, qua_d, zigzag_d, Qua_C_d, Y_size);
	//----------------------------------------------------------------------------
	// ハフマン符号化
	//============================================================================
	gpu_huffman_mcu<<<Dg3_0, Db3_0>>>(qua_d, GPUmOBSP.device_data(), buf_d.getWriteBufAddress(), buf_d.getEndOfBuf(), width, height);

	// 逐次処理のためCPUに戻す
	GPUmOBSP.syncHostMemory();
	cpu_huffman_middle(GPUmOBSP.host_data(), width, height, dst_NumBits.data());
	GPUmOBSP.syncDeviceMemory();

	gpu_huffman_write_devide0<<<Dg3_1, Db3_1>>>(GPUmOBSP.device_data(), buf_d.getWriteBufAddress(), mHeadOfBufP_d, width, height);
	gpu_huffman_write_devide1<<<Dg3_1, Db3_1>>>(GPUmOBSP.device_data(), buf_d.getWriteBufAddress(), mHeadOfBufP_d, width, height);
	gpu_huffman_write_devide2<<<Dg3_1, Db3_1>>>(GPUmOBSP.device_data(), buf_d.getWriteBufAddress(), mHeadOfBufP_d, width, height);

	//----------------------------------------------------------------------------
	// 結果メモリ転送 :出力は「dst_dataとdst_NumBits」の２つ
	//============================================================================
	int dst_size = GPUmOBSP[YCC_size / 64 - 1].mBytePos + (GPUmOBSP[YCC_size / 64 - 1].mBitPos == 7 ? 0 : 1);
	ByteBuffer dst_data(dst_size);
	cudaMemcpy(dst_data.data(), mHeadOfBufP_d, sizeof(byte) * dst_size, cudaMemcpyDeviceToHost);

	//----------------------------------------------------------------------------
	// Decode
	//============================================================================
	//----------------------------------------------------------------------------
	// メモリ確保
	//============================================================================
	InBitStream *mIBSP = new InBitStream(dst_data.data(), dst_size);
	IntBuffer c_qua(width * height + 2 * (width / 2) * (height / 2));

	//並列展開するためのサイズ情報を入れるための構造体
	GPUInBitStream *GPUmIBSP = new GPUInBitStream[YCC_size / 64]; //
	GPUInBitStream *GPUmIBSP_d;
	cudaMalloc((void**) &GPUmIBSP_d, sizeof(GPUInBitStream) * (YCC_size / 64));
	cudaMemcpy(GPUmIBSP_d, GPUmIBSP, sizeof(GPUInBitStream) * (YCC_size / 64), cudaMemcpyHostToDevice);

	//生データを入れるためのバッファ（全体バッファ）
	GPUCInBitStream_BufP IbufP;
	cudaMalloc((void**) &(IbufP.mHeadOfBufP), dst_size);
	cudaMemset(IbufP.mHeadOfBufP, 0, dst_size);

	//----------------------------------------------------------------------------
	// ハフマン復号
	//============================================================================
	// CPU
	decode_huffman(mIBSP, c_qua.data(), width, height);
	// GPU:GPUInstream.hにバグがある可能性もあるので留意
	cudaMemcpy(qua_d, c_qua.data(), sizeof(int) * (width * height + 2 * (width / 2) * (height / 2)),
		cudaMemcpyHostToDevice);

	//----------------------------------------------------------------------------
	// 逆量子化
	//============================================================================
	gpu_izig_quantize_Y<<<Dg2_0, Db2_0>>>(qua_d, coef_d, zigzag_d, Qua_Y_d);
	gpu_izig_quantize_C<<<Dg2_1, Db2_1>>>(qua_d, coef_d, zigzag_d, Qua_C_d, Y_size);

	//----------------------------------------------------------------------------
	// 逆DCT
	//============================================================================
	gpu_idct_0<<<Dg1, Db1>>>(coef_d, f_d, ICosT_d);
	gpu_idct_1<<<Dg1, Db1>>>(f_d, ycc_d, ICosT_d);

	//----------------------------------------------------------------------------
	// yuv->RGB
	//============================================================================
	gpu_color_itrans<<<Dg0_0, Db0_0>>>(ycc_d, src_d, itrans_table_Y.device_data(), itrans_table_C.device_data(), C_size);

	//----------------------------------------------------------------------------
	// 結果転送
	//============================================================================
	cudaMemcpy((byte*) result.getRawData(), src_d, sizeof(byte) * width * height * 3, cudaMemcpyDeviceToHost);

	out_file_name = "gpu_" + out_file_name;
	result.saveToFile(out_file_name);

	//----------------------------------------------------------------------------
	// 開放処理
	//============================================================================
	cudaFree(CosT_d);
	cudaFree(Qua_Y_d);
	cudaFree(Qua_C_d);
	cudaFree(zigzag_d);

	cudaFree(ICosT_d);

	cudaFree(src_d);
	cudaFree(ycc_d);
	cudaFree(coef_d);
	cudaFree(f_d);
	cudaFree(qua_d);
	cudaFree(mHeadOfBufP_d);

	delete mIBSP;
}

