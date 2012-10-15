#include <cstdio>
#include <cstdlib>
#include <cmath>//cos
#include <cstring>//memcpy

#include "cpu_jpeg.h"

#include "utils/util_cv.h"
#include "utils/encoder_tables.h"
#include "utils/out_bit_stream.h"
#include "utils/in_bit_stream.h"

const double kSqrt2 = 1.41421356;			// 2の平方根
const double kDisSqrt2 = 1.0 / 1.41421356;		// 2の平方根の逆数
const double kPaiDiv16 = 3.14159265 / 16;	// 円周率/16

//for DCT
static double CosT[8][8][8][8];
class DCTCoefficient {
public:
	DCTCoefficient() {
		for (int v = 0; v < 8; v++) {
			for (int u = 0; u < 8; u++) {
				for (int y = 0; y < 8; y++) {
					for (int x = 0; x < 8; x++) {
						CosT[u][x][v][y] = cos((2 * x + 1) * u * kPaiDiv16) * cos((2 * y + 1) * v * kPaiDiv16);
					}
				}
			}
		}
	}
} init;

byte revise_value(double v) {
	if (v < 0.0)
		return 0;
	if (v > 255.0)
		return 255;
	return (byte) v;
}

void color_trans_rgb_to_yuv(byte* src_img, int* dst_img, int sizeX, int sizeY) {
	int i, j, k, l, m;
	int src_offset, dst_offset, src_posi, dst_posi;
	int MCU_x = sizeX / 16, MCU_y = sizeY / 16;

	//Y
	for (j = 0; j < MCU_y; j++) {
		for (i = 0; i < MCU_x; i++) {
			for (k = 0; k < 4; k++) {
				switch (k) {
				case 0: //hidariue
					src_offset = 0;
					dst_offset = 0;
					break;
				case 1: //migiue
					src_offset = 8;
					dst_offset = 64;
					break;
				case 2: //hidarishita
					src_offset = 8 * sizeX;
					dst_offset = 128;
					break;
				case 3: //migishita
					src_offset = 8 * sizeX + 8;
					dst_offset = 192;
					break;
				default:
					break;
				}
				for (l = 0; l < 8; l++) {
					for (m = 0; m < 8; m++) {
						src_posi = 3 * (16 * i + 16 * sizeX * j + src_offset + l * sizeX + m);
						dst_posi = 256 * (i + j * MCU_x) + dst_offset + 8 * l + m;

						//Y: -128 levelshift
						dst_img[dst_posi] = int(
							0.1440 * src_img[src_posi + 0] + 0.5870 * src_img[src_posi + 1]
								+ 0.2990 * src_img[src_posi + 2] - 128);
					}
				}

			}

		}

	}

	//CC
	for (j = 0; j < MCU_y; j++) {
		for (i = 0; i < MCU_x; i++) {
			for (l = 0; l < 16; l += 2) {
				for (m = 0; m < 16; m += 2) {
					src_posi = 3 * (16 * i + 16 * sizeX * j + l * sizeX + m);
					dst_posi = (sizeX * sizeY) + 64 * (i + j * MCU_x) + 8 * (l / 2) + (m / 2);
					//Cb
					dst_img[dst_posi] = int(
						0.5000 * src_img[src_posi + 0] - 0.3313 * src_img[src_posi + 1]
							- 0.1687 * src_img[src_posi + 2]);
					//Cr
					dst_img[dst_posi + ((sizeX / 2) * (sizeY / 2))] = int(
						-0.0813 * src_img[src_posi + 0] - 0.4187 * src_img[src_posi + 1]
							+ 0.5000 * src_img[src_posi + 2]);
				}
			}
		}
	}
}

void color_trans_yuv_to_rgb(int *src_img, byte *dst_img, int sizeX, int sizeY) {
	int i, j, k, l, m;
	int src_offset, dst_offset, src_posi, dst_posi;
	int Cb, Cr;
	int MCU_x = sizeX / 16, MCU_y = sizeY / 16;
	int Y_size = sizeX * sizeY, C_size = Y_size / 4; //(sizeX/2)*(sizeY/2)
	//Y
	for (j = 0; j < MCU_y; j++) {
		for (i = 0; i < MCU_x; i++) {
			for (k = 0; k < 4; k++) {
				switch (k) {
				case 0:
					src_offset = 0;
					dst_offset = 0;
					break;
				case 1:
					src_offset = 64;
					dst_offset = 8;
					break;
				case 2:
					src_offset = 128;
					dst_offset = 8 * sizeX;
					break;
				case 3:
					src_offset = 192;
					dst_offset = 8 * sizeX + 8;
					break;
				default:
					//printf("unanticipated k");
					break;
				}
				for (l = 0; l < 8; l++) { //tate
					for (m = 0; m < 8; m++) { //yoko
						src_posi = 256 * (i + j * MCU_x) + src_offset + 8 * l + m;

						Cb = Y_size + 64 * (i + j * MCU_x) + ksamplingT[src_offset + 8 * l + m];
						Cr = Cb + C_size;

						dst_posi = 3 * (16 * i + 16 * sizeX * j + dst_offset + sizeX * l + m);

						//BGR
						dst_img[dst_posi] = revise_value(src_img[src_posi] + 1.77200 * (src_img[Cb] - 128));
						dst_img[dst_posi + 1] = revise_value(
							src_img[src_posi] - 0.34414 * (src_img[Cb] - 128) - 0.71414 * (src_img[Cr] - 128));
						dst_img[dst_posi + 2] = revise_value(src_img[src_posi] + 1.40200 * (src_img[Cr] - 128));

					}
				}
			}
		}
	}
}

void dct(int *src_ycc, int *dst_coef, int sizeX, int sizeY) {
	int v, u, y, x;
	double cv, cu, sum;
	const int size = sizeX * sizeY + 2 * (sizeX / 2) * (sizeY / 2);
	//DCT
	for (int i = 0; i < size; i += 64) {
		for (v = 0; v < 8; v++) {
			cv = v ? 1.0 : kDisSqrt2;
			for (u = 0; u < 8; u++) {
				cu = u ? 1.0 : kDisSqrt2;
				sum = 0;
				for (y = 0; y < 8; y++) {
					for (x = 0; x < 8; x++) {
						sum += src_ycc[i + y * 8 + x] * CosT[u][x][v][y];
					}
				}
				dst_coef[i + v * 8 + u] = int(sum * cu * cv / 4);
			}
		}
	}
}

void idct(int *src_coef, int *dst_ycc, int sizeX, int sizeY) {
	int v, u, y, x;
	double cv, cu, sum;
	const int size = sizeX * sizeY + 2 * (sizeX / 2) * (sizeY / 2);
	//IDCT
	for (int i = 0; i < size; i += 64) {
		for (y = 0; y < 8; y++) {
			for (x = 0; x < 8; x++) {
				sum = 0;
				for (v = 0; v < 8; v++) {
					cv = v == 0 ? kDisSqrt2 : 1.0;
					for (u = 0; u < 8; u++) {
						cu = u == 0 ? kDisSqrt2 : 1.0;
						sum += cu * cv * src_coef[i + v * 8 + u] * CosT[u][x][v][y];
					}
				}
				dst_ycc[i + y * 8 + x] = int(sum / 4 + 128);
			}
		}
	}
}

void zig_quantize(int *src_coef, int *dst_qua, int sizeX, int sizeY) {
	int i;
	const int Ysize = sizeX * sizeY;
	const int Csize = (sizeX / 2) * (sizeY / 2);
	//Y
	for (i = 0; i < Ysize; i++) {
		dst_qua[64 * (i / 64) + kZigzagT[i % 64]] = src_coef[i] / kYQuantumT[i % 64];
	}
	//C(Cb,Cr)
	for (i = Ysize; i < 2 * Csize; i++) {
		dst_qua[64 * (i / 64) + kZigzagT[i % 64]] = src_coef[i] / kCQuantumT[i % 64];
	}

}

void izig_quantize(int *src_qua, int *dst_coef, int sizeX, int sizeY) {
	int i;
	const int Ysize = sizeX * sizeY;
	const int Csize = (sizeX / 2) * (sizeY / 2);
	//Y
	for (i = 0; i < Ysize; i++) {
		dst_coef[i] = src_qua[64 * (i / 64) + kZigzagT[i % 64]] * kYQuantumT[i % 64];
	}
	//C(Cb,Cr)
	for (i = Ysize; i < 2 * Csize; i++) {
		dst_coef[i] = src_qua[64 * (i / 64) + kZigzagT[i % 64]] * kCQuantumT[i % 64];
	}

}

//if文を減らすためやや冗長な書き方をしている
void huffman_encode(int *src_qua, OutBitStream *mOBSP, int sizeX, int sizeY) {
	int i, s, v;
	int preDC = 0, diff;
	int absC;
	int dIdx, aIdx;
	const int Ysize = sizeX * sizeY;
	const int Cbsize = Ysize + sizeX * sizeY / 4; //(size/2)*(size/2)
	const int Crsize = Ysize + sizeX * sizeY / 2; //2*(size/2)*(size/2)

	int run = 0;

	//Y
	for (i = 0; i < Ysize; i++) {
		//DC
		if (i % 64 == 0) {
			diff = src_qua[i] - preDC;
			preDC = src_qua[i];
			absC = abs(diff);
			dIdx = 0;
			while (absC > 0) {
				absC >>= 1;
				dIdx++;
			}
			mOBSP->SetBits((kYDcHuffmanT.CodeTP)[dIdx], (kYDcHuffmanT.SizeTP)[dIdx]);
			if (dIdx) {
				if (diff < 0)
					diff--;
				mOBSP->SetBits(diff, dIdx);
			}
			run = 0;
		}
		//AC
		else {
			absC = abs(src_qua[i]);
			if (absC) {
				while (run > 15) {
					mOBSP->SetBits((kYAcHuffmanT.CodeTP)[kYZRLidx], (kYAcHuffmanT.SizeTP)[kYZRLidx]);
					run -= 16;
				}
				s = 0;
				while (absC > 0) {
					absC >>= 1;
					s++;
				}
				aIdx = run * 10 + s + (run == 15);
				mOBSP->SetBits((kYAcHuffmanT.CodeTP)[aIdx], (kYAcHuffmanT.SizeTP)[aIdx]);
				v = src_qua[i];
				if (v < 0)
					v--;
				mOBSP->SetBits(v, s);

				run = 0;
			} else {
				if (i % 64 == 63)
					mOBSP->SetBits((kYAcHuffmanT.CodeTP)[kYEOBidx], (kYAcHuffmanT.SizeTP)[kYEOBidx]);
				else
					run++;
			}
		}
	}

	//Cb
	preDC = 0;
	run = 0;
	for (i = Ysize; i < Cbsize; i++) { //Cb,Cr
		//DC
		if (i % 64 == 0) {
			diff = src_qua[i] - preDC;
			preDC = src_qua[i];
			absC = abs(diff);
			dIdx = 0;
			while (absC > 0) {
				absC >>= 1;
				dIdx++;
			}
			mOBSP->SetBits((kCDcHuffmanT.CodeTP)[dIdx], (kCDcHuffmanT.SizeTP)[dIdx]);
			if (dIdx) {
				if (diff < 0)
					diff--;
				mOBSP->SetBits(diff, dIdx);
			}
			run = 0;
		}
		//AC
		else {
			absC = abs(src_qua[i]);
			if (absC) {
				while (run > 15) {
					mOBSP->SetBits((kCAcHuffmanT.CodeTP)[kCZRLidx], (kCAcHuffmanT.SizeTP)[kCZRLidx]);
					run -= 16;
				}
				s = 0;
				while (absC > 0) {
					absC >>= 1;
					s++;
				}
				aIdx = run * 10 + s + (run == 15);
				mOBSP->SetBits((kCAcHuffmanT.CodeTP)[aIdx], (kCAcHuffmanT.SizeTP)[aIdx]);
				v = src_qua[i];
				if (v < 0)
					v--;
				mOBSP->SetBits(v, s);

				run = 0;
			} else {
				if (i % 64 == 63)
					mOBSP->SetBits((kCAcHuffmanT.CodeTP)[kCEOBidx], (kCAcHuffmanT.SizeTP)[kCEOBidx]);
				else
					run++;
			}
		}
	}
	//Cr
	preDC = 0;
	run = 0;
	for (i = Cbsize; i < Crsize; i++) { //Cb,Cr
		//DC
		if (i % 64 == 0) {
			diff = src_qua[i] - preDC;
			preDC = src_qua[i];
			absC = abs(diff);
			dIdx = 0;
			while (absC > 0) {
				absC >>= 1;
				dIdx++;
			}
			mOBSP->SetBits((kCDcHuffmanT.CodeTP)[dIdx], (kCDcHuffmanT.SizeTP)[dIdx]);
			if (dIdx) {
				if (diff < 0)
					diff--;
				mOBSP->SetBits(diff, dIdx);
			}
			run = 0;
		}
		//AC
		else {
			absC = abs(src_qua[i]);
			if (absC) {
				while (run > 15) {
					mOBSP->SetBits((kCAcHuffmanT.CodeTP)[kCZRLidx], (kCAcHuffmanT.SizeTP)[kCZRLidx]);
					run -= 16;
				}
				s = 0;
				while (absC > 0) {
					absC >>= 1;
					s++;
				}
				aIdx = run * 10 + s + (run == 15);
				mOBSP->SetBits((kCAcHuffmanT.CodeTP)[aIdx], (kCAcHuffmanT.SizeTP)[aIdx]);
				v = src_qua[i];
				if (v < 0)
					v--;
				mOBSP->SetBits(v, s);

				run = 0;
			} else {
				if (i % 64 == 63)
					mOBSP->SetBits((kCAcHuffmanT.CodeTP)[kCEOBidx], (kCAcHuffmanT.SizeTP)[kCEOBidx]);
				else
					run++;
			}
		}
	}
}

// ハフマン符号1語の復号
int decode_huffman_word(InBitStream *mIBSP, int tc, int sc) { //sc:Y==0,C==1,tc:DC==0,AC==1
	// ハフマンテーブル指定
	const SHuffmanDecodeTable &theHT = (
		sc == 0 ? (tc == 0 ? kYDcHuffmanDT : kYAcHuffmanDT) : (tc == 0 ? kCDcHuffmanDT : kCAcHuffmanDT)); // 使用するハフマンテーブル

	int code = 0; // ハフマン符号語の候補：最大値16ビット
	int length = 0; // ハフマン符号語候補のビット数
	int next = 0; // 次の1ビット
	int k = 0; // 表の指数

	while (k < theHT.numOfElement && length < 16) {
		length++;
		code <<= 1;
		next = mIBSP->GetBits(1);
		//if( next < 0 )// マーカだったら
		//  return next;
		code |= next;

		while ((theHT.SizeTP)[k] == length) { // 候補と符号語のビット数が等しい間だ検索
			if ((theHT.CodeTP)[k] == code) { // ヒット
				return (theHT.ValueTP)[k]; // 復号結果を返す
			}
			k++; // 次の符号語
		}
	}
	return 0;
}

void decode_huffman(InBitStream *mIBSP, int *dst_qua, int sizeX, int sizeY) {
	int i, k;
	int preDC = 0, diff;
	int category;

	const int Y_MCU_num = sizeX * sizeY / 64;
	const int Cb_MCU_num = Y_MCU_num + ((sizeX / 2) * (sizeY / 2)) / 64;
	const int Cr_MCU_num = Cb_MCU_num + ((sizeX / 2) * (sizeY / 2)) / 64;
	int run = 0, acv = 0;

	////// Y //////////
	for (i = 0; i < Y_MCU_num; i++) { //MCU
		//DC
		diff = 0;
		category = decode_huffman_word(mIBSP, 0, 0);

		diff = mIBSP->GetBits(category);
		if ((diff & (1 << (category - 1))) == 0) { //負
			diff -= (1 << category) - 1;
		}
		//}
		preDC += diff;
		dst_qua[i * 64] = preDC;

		//AC
		k = 1;
		while (k < 64) {
			category = decode_huffman_word(mIBSP, 1, 0);
			if (category == 0) { //EOB
				while (k < 64) {
					dst_qua[i * 64 + (k++)] = 0;
				}
				break;
			}

			run = category >> 4; //run length
			category &= 0x0f; //category
			acv = 0;
			if (category) {
				acv = mIBSP->GetBits(category);
				if ((acv & (1 << (category - 1))) == 0)
					acv -= (1 << category) - 1; //負
			}

			while (run-- > 0) { //ランレングスの数だけ0
				dst_qua[i * 64 + (k++)] = 0;
			}
			dst_qua[i * 64 + (k++)] = acv;

		}
	}

	////// Cb /////////////
	preDC = 0;
	for (i = Y_MCU_num; i < Cb_MCU_num; i++) { //Cb,Cr
		diff = 0;
		category = decode_huffman_word(mIBSP, 0, 1);

		diff = mIBSP->GetBits(category);
		if ((diff & (1 << (category - 1))) == 0) { //負
			diff -= (1 << category) - 1;
		}
		//}
		preDC += diff;
		dst_qua[i * 64] = preDC;

		//AC
		k = 1;
		while (k < 64) {
			category = decode_huffman_word(mIBSP, 1, 1);
			if (category == 0) { //EOB
				while (k < 64) {
					dst_qua[i * 64 + (k++)] = 0;
				}
				break;
			}

			run = category >> 4; //run length
			category &= 0x0f; //category
			acv = 0;
			if (category) {
				acv = mIBSP->GetBits(category);
				if ((acv & (1 << (category - 1))) == 0)
					acv -= (1 << category) - 1; //負
			}

			while (run-- > 0) { //ランレングスの数だけ0
				dst_qua[i * 64 + (k++)] = 0;
			}
			dst_qua[i * 64 + (k++)] = acv;

		}
	}
	////// Cr /////////////
	preDC = 0;
	for (i = Cb_MCU_num; i < Cr_MCU_num; i++) { //Cb,Cr
		//DC
		diff = 0;
		category = decode_huffman_word(mIBSP, 0, 1);

		diff = mIBSP->GetBits(category);
		if ((diff & (1 << (category - 1))) == 0) { //負
			diff -= (1 << category) - 1;
		}
		//}
		preDC += diff;
		dst_qua[i * 64] = preDC;

		//AC
		k = 1;
		while (k < 64) {
			category = decode_huffman_word(mIBSP, 1, 1);
			if (category == 0) { //EOB
				while (k < 64) {
					dst_qua[i * 64 + (k++)] = 0;
				}
				//continue;
				break;
			}

			run = category >> 4; //run length
			category &= 0x0f; //category
			acv = 0;
			if (category) {
				acv = mIBSP->GetBits(category);
				if ((acv & (1 << (category - 1))) == 0)
					acv -= (1 << category) - 1; //負
			}

			while (run-- > 0) { //ランレングスの数だけ0
				dst_qua[i * 64 + (k++)] = 0;
			}
			dst_qua[i * 64 + (k++)] = acv;

		}
	}
}
