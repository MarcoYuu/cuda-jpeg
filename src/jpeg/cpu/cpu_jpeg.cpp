﻿#include <cstdio>
#include <cstdlib>
#include <cmath>//cos
#include <cstring>//memcpy
#include <cassert>

#include <string>

#include <jpeg/cpu/cpu_jpeg.h>

#include <utils/util_cv.h>
#include <utils/out_bit_stream.h>
#include <utils/in_bit_stream.h>

#include "encoder_tables.h"

using namespace util;
using namespace jpeg::cpu::encode_table;

namespace jpeg {
	namespace cpu {
		// -------------------------------------------------------------------------
		// 符号化に用いる定数
		// =========================================================================
		static const double kDisSqrt2 = 1.0 / 1.41421356;	 	//! 2の平方根の逆数
		static const double kPaiDiv16 = 3.14159265 / 16; 		//! 円周率/16

		static double CosT[8][8][8][8];
		/**
		 * @brief DCT用行列初期化クラス
		 *
		 * @author yuumomma
		 * @version 1.0
		 */
		static class DCTMatrix {
		public:
			DCTMatrix() {
				for (int v = 0; v < 8; v++) {
					for (int u = 0; u < 8; u++) {
						for (int y = 0; y < 8; y++) {
							for (int x = 0; x < 8; x++) {
								CosT[u][x][v][y] = cos((2 * x + 1) * u * kPaiDiv16)
									* cos((2 * y + 1) * v * kPaiDiv16);
							}
						}
					}
				}
			}
		} init;

		// -------------------------------------------------------------------------
		// CPUエンコーダ
		// =========================================================================
		JpegEncoder::JpegEncoder() :
			yuv_data_(),
			coefficient_(),
			quantized_(),
			out_bit_(1),
			width_(0),
			height_(0) {

		}

		JpegEncoder::JpegEncoder(size_t width, size_t height) :
			yuv_data_((size_t) (width * height * 3 / 2)),
			coefficient_((size_t) (width * height * 3 / 2)),
			quantized_((size_t) (width * height * 3 / 2)),
			out_bit_((size_t) (width * height * 3)),
			width_(width),
			height_(height) {

		}

		void JpegEncoder::reset() {
			out_bit_.reset();
		}

		void JpegEncoder::setImageSize(size_t width, size_t height) {
			yuv_data_.resize((size_t) (width * height * 3 / 2));
			coefficient_.resize((size_t) (width * height * 3 / 2));
			quantized_.resize((size_t) (width * height * 3 / 2));

			out_bit_.resize((size_t) (width * height * 3));

			width_ = width;
			height_ = height;
		}

		size_t JpegEncoder::encode(const byte* rgb_data, size_t src_size, byte* result, size_t result_size) {
			assert(src_size == width_ * height_ * 3);
			inner_encode(rgb_data);

			size_t size = out_bit_.getStreamSize();
			assert(result_size >= size);

			memcpy(result, out_bit_.getStreamAddress(), size);
			reset();

			return size;
		}

		size_t JpegEncoder::encode(const ByteBuffer &rgb_data, byte *result, size_t result_size) {
			inner_encode(rgb_data.data());

			size_t size = out_bit_.getStreamSize();
			memcpy(result, out_bit_.getStreamAddress(), size);
			reset();

			return size;
		}

		size_t JpegEncoder::encode(const byte *rgb_data, size_t src_size, ByteBuffer &result) {
			assert(src_size == width_ * height_ * 3);

			inner_encode(rgb_data);

			size_t size = out_bit_.getStreamSize();
			result.assign(out_bit_.getStreamAddress(), out_bit_.getStreamAddress() + size);
			reset();

			return size;
		}

		size_t JpegEncoder::encode(const ByteBuffer &rgb_data, ByteBuffer &result) {
			inner_encode(rgb_data.data());

			size_t size = out_bit_.getStreamSize();
			result.assign(out_bit_.getStreamAddress(), out_bit_.getStreamAddress() + size);
			reset();

			return size;
		}

		void JpegEncoder::inner_encode(const byte* rgb_data) {
			color_trans_rgb_to_yuv(rgb_data, yuv_data_.data(), width_, height_);
			dct(yuv_data_.data(), coefficient_.data(), width_, height_);
			zig_quantize(coefficient_.data(), quantized_.data(), width_, height_);
			encode_huffman(quantized_.data(), &out_bit_, width_, height_);
		}

		// -------------------------------------------------------------------------
		// CPUデコーダ
		// =========================================================================
		JpegDecoder::JpegDecoder() :
			yuv_data_(),
			coefficient_(),
			quantized_(),
			width_(0),
			height_(0) {

		}

		JpegDecoder::JpegDecoder(size_t width, size_t height) :
			yuv_data_(width * height * 3 / 2),
			coefficient_(width * height * 3 / 2),
			quantized_(width * height * 3 / 2),
			width_(width),
			height_(height) {
		}

		void JpegDecoder::setImageSize(size_t width, size_t height) {
			yuv_data_.resize(width * height * 3 / 2);
			coefficient_.resize(width * height * 3 / 2);
			quantized_.resize(width * height * 3 / 2);
			width_ = width;
			height_ = height;
		}

		void JpegDecoder::decode(const byte *src, size_t src_size, byte *result, size_t result_size) {
			assert(result_size>=width_ * height_ * 3);
			InBitStream in_bit(src, src_size);

			inner_decode(&in_bit, result);
		}

		void JpegDecoder::decode(const ByteBuffer& src, ByteBuffer &result) {
			result.resize(width_ * height_ * 3);
			InBitStream in_bit(src.data(), src.size());

			inner_decode(&in_bit, result.data());
		}

		void JpegDecoder::decode(const byte *src, size_t src_size, ByteBuffer &result) {
			result.resize(width_ * height_ * 3);
			InBitStream in_bit(src, src_size);

			inner_decode(&in_bit, result.data());
		}

		void JpegDecoder::decode(const ByteBuffer &src, byte *result, size_t result_size) {
			assert(result_size>=width_ * height_ * 3);
			InBitStream in_bit(src.data(), src.size());

			inner_decode(&in_bit, result);
		}

		void JpegDecoder::inner_decode(InBitStream *in_bit, byte *result) {
			decode_huffman(in_bit, quantized_.data(), width_, height_);
			izig_quantize(quantized_.data(), coefficient_.data(), width_, height_);
			idct(coefficient_.data(), yuv_data_.data(), width_, height_);
			color_trans_yuv_to_rgb(yuv_data_.data(), result, width_, height_);
		}

		// -------------------------------------------------------------------------
		// 符号化関数
		// =========================================================================
		byte revise_value(double v) {
			if (v < 0.0)
				return 0;
			if (v > 255.0)
				return 255;
			return (byte) v;
		}

		void color_trans_rgb_to_yuv(const byte* src_img, int* dst_img, size_t sizeX, size_t sizeY) {
			int i, j, k, l, m;
			size_t src_offset, dst_offset, src_posi, dst_posi;
			size_t MCU_x = sizeX / 16, MCU_y = sizeY / 16;

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

		void color_trans_yuv_to_rgb(const int *src_img, byte *dst_img, size_t sizeX, size_t sizeY) {
			int i, j, k, l, m;
			size_t src_offset, dst_offset, src_posi, dst_posi;
			size_t Cb, Cr;
			size_t MCU_x = sizeX / 16, MCU_y = sizeY / 16;
			size_t Y_size = sizeX * sizeY, C_size = Y_size / 4; //(sizeX/2)*(sizeY/2)
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

								Cb = Y_size + 64 * (i + j * MCU_x)
									+ Sampling::luminance[src_offset + 8 * l + m];
								Cr = Cb + C_size;

								dst_posi = 3 * (16 * i + 16 * sizeX * j + dst_offset + sizeX * l + m);

								//BGR
								dst_img[dst_posi] = revise_value(
									src_img[src_posi] + 1.77200 * (src_img[Cb] - 128));
								dst_img[dst_posi + 1] = revise_value(
									src_img[src_posi] - 0.34414 * (src_img[Cb] - 128)
										- 0.71414 * (src_img[Cr] - 128));
								dst_img[dst_posi + 2] = revise_value(
									src_img[src_posi] + 1.40200 * (src_img[Cr] - 128));

							}
						}
					}
				}
			}
		}

		void dct(const int *src_ycc, int *dst_coef, size_t sizeX, size_t sizeY) {
			int v, u, y, x;
			double cv, cu, sum;
			const size_t size = sizeX * sizeY + 2 * (sizeX / 2) * (sizeY / 2);
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

		void idct(const int *src_coef, int *dst_ycc, size_t sizeX, size_t sizeY) {
			int v, u, y, x;
			double cv, cu, sum;
			const size_t size = sizeX * sizeY + 2 * (sizeX / 2) * (sizeY / 2);
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

		void zig_quantize(const int *src_coef, int *dst_qua, size_t sizeX, size_t sizeY) {
			const size_t Ysize = sizeX * sizeY;
			const size_t Csize = (sizeX / 2) * (sizeY / 2);

			//Y
			for (size_t i = 0; i < Ysize; i++) {
				dst_qua[64 * (i / 64) + Zigzag::sequence[i % 64]] = src_coef[i] / Quantize::luminance[i % 64];
			}
			//C(Cb,Cr)
			for (size_t i = Ysize; i < Ysize + 2 * Csize; i++) {
				dst_qua[64 * (i / 64) + Zigzag::sequence[i % 64]] = src_coef[i] / Quantize::component[i % 64];
			}
		}

		void izig_quantize(const int *src_qua, int *dst_coef, size_t sizeX, size_t sizeY) {
			const size_t Ysize = sizeX * sizeY;
			const size_t Csize = (sizeX / 2) * (sizeY / 2);

			//Y
			for (size_t i = 0; i < Ysize; i++) {
				dst_coef[i] = src_qua[64 * (i / 64) + Zigzag::sequence[i % 64]] * Quantize::luminance[i % 64];
			}
			//C(Cb,Cr)
			for (size_t i = Ysize; i < Ysize + 2 * Csize; i++) {
				dst_coef[i] = src_qua[64 * (i / 64) + Zigzag::sequence[i % 64]] * Quantize::component[i % 64];
			}
		}

		//if文を減らすためやや冗長な書き方をしている
		void encode_huffman(const int *src_qua, OutBitStream *obit_stream, size_t sizeX, size_t sizeY) {
			const size_t Ysize = sizeX * sizeY;
			const size_t Cbsize = Ysize + sizeX * sizeY / 4; //(size/2)*(size/2)
			const size_t Crsize = Ysize + sizeX * sizeY / 2; //2*(size/2)*(size/2)

			//Y
			int run = 0;
			int preDC = 0;
			for (int i = 0; i < Ysize; i++) {
				//DC
				if (i % 64 == 0) {
					using HuffmanEncode::DC::luminance;

					int diff = src_qua[i] - preDC;
					preDC = src_qua[i];
					int absC = abs(diff);
					int dIdx = 0;
					while (absC > 0) {
						absC >>= 1;
						dIdx++;
					}
					obit_stream->setBits(luminance::code[dIdx], luminance::size[dIdx]);
					if (dIdx) {
						if (diff < 0)
							diff--;
						obit_stream->setBits(diff, dIdx);
					}
					run = 0;
				}
				//AC
				else {
					using HuffmanEncode::AC::luminance;
					int absC = abs(src_qua[i]);
					if (absC) {
						while (run > 15) {
							obit_stream->setBits(luminance::code[luminance::ZRL],
								luminance::size[luminance::ZRL]);
							run -= 16;
						}
						int s = 0;
						while (absC > 0) {
							absC >>= 1;
							s++;
						}
						int aIdx = run * 10 + s + (run == 15);
						obit_stream->setBits(luminance::code[aIdx], luminance::size[aIdx]);
						int v = src_qua[i];
						if (v < 0)
							v--;
						obit_stream->setBits(v, s);

						run = 0;
					} else {
						if (i % 64 == 63)
							obit_stream->setBits(luminance::code[luminance::EOB],
								luminance::size[luminance::EOB]);
						else
							run++;
					}
				}
			}

			//Cb
			preDC = 0;
			run = 0;
			for (size_t i = Ysize; i < Cbsize; i++) { //Cb,Cr
				//DC
				if (i % 64 == 0) {
					using HuffmanEncode::DC::component;
					int diff = src_qua[i] - preDC;
					preDC = src_qua[i];
					int absC = abs(diff);
					int dIdx = 0;
					while (absC > 0) {
						absC >>= 1;
						dIdx++;
					}
					obit_stream->setBits(component::code[dIdx], component::size[dIdx]);
					if (dIdx) {
						if (diff < 0)
							diff--;
						obit_stream->setBits(diff, dIdx);
					}
					run = 0;
				}
				//AC
				else {
					using HuffmanEncode::AC::component;
					int absC = abs(src_qua[i]);
					if (absC) {
						while (run > 15) {
							obit_stream->setBits(component::code[component::ZRL],
								component::size[component::ZRL]);
							run -= 16;
						}
						int s = 0;
						while (absC > 0) {
							absC >>= 1;
							s++;
						}
						int aIdx = run * 10 + s + (run == 15);
						obit_stream->setBits(component::code[aIdx], component::size[aIdx]);
						int v = src_qua[i];
						if (v < 0)
							v--;
						obit_stream->setBits(v, s);

						run = 0;
					} else {
						if (i % 64 == 63)
							obit_stream->setBits(component::code[component::EOB],
								component::size[component::EOB]);
						else
							run++;
					}
				}
			}
			//Cr
			preDC = 0;
			run = 0;
			for (size_t i = Cbsize; i < Crsize; i++) { //Cb,Cr
				//DC
				if (i % 64 == 0) {
					using HuffmanEncode::DC::component;

					int diff = src_qua[i] - preDC;
					preDC = src_qua[i];
					int absC = abs(diff);
					int dIdx = 0;
					while (absC > 0) {
						absC >>= 1;
						dIdx++;
					}
					obit_stream->setBits(component::code[dIdx], component::size[dIdx]);
					if (dIdx) {
						if (diff < 0)
							diff--;
						obit_stream->setBits(diff, dIdx);
					}
					run = 0;
				}
				//AC
				else {
					using HuffmanEncode::AC::component;

					int absC = abs(src_qua[i]);
					if (absC) {
						while (run > 15) {
							obit_stream->setBits(component::code[component::ZRL],
								component::size[component::ZRL]);
							run -= 16;
						}
						int s = 0;
						while (absC > 0) {
							absC >>= 1;
							s++;
						}
						int aIdx = run * 10 + s + (run == 15);
						obit_stream->setBits(component::code[aIdx], component::size[aIdx]);
						int v = src_qua[i];
						if (v < 0)
							v--;
						obit_stream->setBits(v, s);

						run = 0;
					} else {
						if (i % 64 == 63)
							obit_stream->setBits(component::code[component::EOB],
								component::size[component::EOB]);
						else
							run++;
					}
				}
			}
		}

		// ハフマン符号1語の復号
		int decode_huffman_word(InBitStream *ibit_stream, int tc, int sc) {		//tc:DC==0,AC==1, sc:Y==0,C==1,
			// ハフマンテーブル指定
			using HuffmanDecode::TableSet;
			const TableSet &theHT = (
				sc == 0 ? (tc == 0 ? TableSet::luminanceDC : TableSet::luminanceAC) :
							(tc == 0 ? TableSet::componentDC : TableSet::componentAC));		// 使用するハフマンテーブル

			int code = 0; // ハフマン符号語の候補：最大値16ビット
			int length = 0; // ハフマン符号語候補のビット数
			int next = 0; // 次の1ビット
			int k = 0; // 表の指数

			while (k < theHT.table_size && length < 16) {
				length++;
				code <<= 1;
				next = ibit_stream->getBits(1);

				code |= next;

				while (theHT.size_table[k] == length) { // 候補と符号語のビット数が等しい間検索
					if (theHT.code_table[k] == code) { // ヒット
						return theHT.value_table[k]; // 復号結果を返す
					}
					k++; // 次の符号語
				}
			}
			return 0;
		}

		void decode_huffman(InBitStream *ibit_stream, int *dst_qua, size_t sizeX, size_t sizeY) {
			const size_t Y_MCU_num = sizeX * sizeY / 64;
			const size_t Cb_MCU_num = Y_MCU_num + ((sizeX / 2) * (sizeY / 2)) / 64;
			const size_t Cr_MCU_num = Cb_MCU_num + ((sizeX / 2) * (sizeY / 2)) / 64;

			////// Y //////////
			int preDC = 0;
			for (int i = 0; i < Y_MCU_num; i++) { //MCU
				//DC
				int diff = 0;
				int category = decode_huffman_word(ibit_stream, 0, 0);

				diff = ibit_stream->getBits(category);
				if ((diff & (1 << (category - 1))) == 0) { //負
					diff -= (1 << category) - 1;
				}
				//}
				preDC += diff;
				dst_qua[i * 64] = preDC;

				//AC
				int k = 1;
				while (k < 64) {
					category = decode_huffman_word(ibit_stream, 1, 0);
					if (category == 0) { //EOB
						while (k < 64) {
							dst_qua[i * 64 + (k++)] = 0;
						}
						break;
					}

					int run = category >> 4; //run length
					category &= 0x0f; //category
					int acv = 0;
					if (category) {
						acv = ibit_stream->getBits(category);
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
			for (size_t i = Y_MCU_num; i < Cb_MCU_num; i++) { //Cb,Cr
				int diff = 0;
				int category = decode_huffman_word(ibit_stream, 0, 1);

				diff = ibit_stream->getBits(category);
				if ((diff & (1 << (category - 1))) == 0) { //負
					diff -= (1 << category) - 1;
				}
				//}
				preDC += diff;
				dst_qua[i * 64] = preDC;

				//AC
				int k = 1;
				while (k < 64) {
					category = decode_huffman_word(ibit_stream, 1, 1);
					if (category == 0) { //EOB
						while (k < 64) {
							dst_qua[i * 64 + (k++)] = 0;
						}
						break;
					}

					int run = category >> 4; //run length
					category &= 0x0f; //category
					int acv = 0;
					if (category) {
						acv = ibit_stream->getBits(category);
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
			for (size_t i = Cb_MCU_num; i < Cr_MCU_num; i++) { //Cb,Cr
				//DC
				int diff = 0;
				int category = decode_huffman_word(ibit_stream, 0, 1);

				diff = ibit_stream->getBits(category);
				if ((diff & (1 << (category - 1))) == 0) { //負
					diff -= (1 << category) - 1;
				}
				//}
				preDC += diff;
				dst_qua[i * 64] = preDC;

				//AC
				int k = 1;
				while (k < 64) {
					category = decode_huffman_word(ibit_stream, 1, 1);
					if (category == 0) { //EOB
						while (k < 64) {
							dst_qua[i * 64 + (k++)] = 0;
						}
						//continue;
						break;
					}

					int run = category >> 4; //run length
					category &= 0x0f; //category
					int acv = 0;
					if (category) {
						acv = ibit_stream->getBits(category);
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
	}  // namespace cpu
}  // namespace jpeg
