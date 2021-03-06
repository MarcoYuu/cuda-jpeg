﻿/******************************************************
EncoderTables Ver.1.1.0 JPEG符号化クラス用テーブル
	Copyright 2004 AsWe.Co.Ltd. All rights reserved.
The comments are written in Japanese (Shift-JIS).
******************************************************/

#ifndef ENCODER_TABLE_DEVICE_H_
#define ENCODER_TABLE_DEVICE_H_

namespace jpeg {
	namespace cuda {
		namespace encode_table {
			//----------------------------------------------------------------------------
			// DCT/iDCT
			//============================================================================
			/**
			 * DCT用定数
			 */
			namespace DCTConstants {
				/** 8x8DCTの変換係数行列 */
				__device__ __constant__ static const float cos[] = {
					0.35355338f,  0.35355338f,  0.35355338f,  0.35355338f,  0.35355338f,  0.35355338f,  0.35355338f,  0.35355338f,
					0.49039263f,  0.41573480f,  0.27778509f,  0.09754512f, -0.09754516f, -0.27778518f, -0.41573483f, -0.49039266f,
					0.46193978f,  0.19134171f, -0.19134176f, -0.46193978f, -0.46193978f, -0.19134156f,  0.19134180f,  0.46193978f,
					0.41573480f, -0.09754516f, -0.49039266f, -0.27778500f,	0.27778521f,  0.49039263f,  0.09754504f, -0.41573489f,
					0.35355338f, -0.35355338f, -0.35355332f,  0.35355350f,  0.35355338f, -0.35355362f, -0.35355327f,  0.35355341f,
					0.27778509f, -0.49039266f,  0.09754521f,  0.41573468f, -0.41573489f, -0.09754511f,  0.49039266f, -0.27778542f,
					0.19134171f, -0.46193978f,  0.46193978f, -0.19134195f, -0.19134149f,  0.46193966f, -0.46193987f,  0.19134195f,
					0.09754512f, -0.27778500f,  0.41573468f, -0.49039260f,  0.49039271f, -0.41573480f,  0.27778557f, -0.09754577f
				 };
				/** 8x8DCTの変換係数転地行列 */
				__device__ __constant__ static const float cos_t[] = {
					0.35355338f,  0.49039263f,  0.46193978f,  0.41573480f,  0.35355338f,  0.27778509f,  0.19134171f,  0.09754512f,
					0.35355338f,  0.41573480f,  0.19134171f, -0.09754516f, -0.35355338f, -0.49039266f, -0.46193978f, -0.27778500f,
					0.35355338f,  0.27778509f, -0.19134176f, -0.49039266f, -0.35355332f,  0.09754521f,  0.46193978f,  0.41573468f,
					0.35355338f,  0.09754512f, -0.46193978f, -0.27778500f,  0.35355350f,  0.41573468f, -0.19134195f, -0.49039260f,
					0.35355338f, -0.09754516f, -0.46193978f,  0.27778521f,  0.35355338f, -0.41573489f, -0.19134149f,  0.49039271f,
					0.35355338f, -0.27778518f, -0.19134156f,  0.49039263f, -0.35355362f, -0.09754511f,  0.46193966f, -0.41573480f,
					0.35355338f, -0.41573483f,  0.19134180f,  0.09754504f, -0.35355327f,  0.49039266f, -0.46193987f,  0.27778557f,
					0.35355338f, -0.49039266f,  0.46193978f, -0.41573489f,  0.35355341f, -0.27778542f,  0.19134195f, -0.09754577f
				 };
			} // namespace DCTConstants

			//----------------------------------------------------------------------------
			// ジグザグシーケンス
			//============================================================================
			namespace Zigzag {
				/** ジグザグシーケンス用 */
				__device__ __constant__ static const u_int	sequence[64] = {
					 0,  1,  5,  6, 14, 15, 27, 28,
					 2,  4,  7, 13, 16, 26, 29, 42,
					 3,  8, 12, 17, 25, 30, 41, 43,
					 9, 11, 18, 24, 31, 40, 44, 53,
					10, 19, 23, 32, 39, 45, 52, 54,
					20, 22, 33, 38, 46, 51, 55, 60,
					21, 34, 37, 47, 50, 56, 59, 61,
					35, 36, 48, 49, 57, 58, 62, 63
				};
			}  // namespace Zigzag

			//----------------------------------------------------------------------------
			// 量子化テーブル
			//============================================================================
			namespace Quantize{
				/**
				* 量子化テーブル輝度用
				* ファイル出力、内部処理兼用
				*/
				__device__ __constant__ static const int	luminance[] = {
					16,  11,  10,  16,  24,  40,  51,  61,
					12,  12,  14,  19,  26,  58,  60,  55,
					14,  13,  16,  24,  40,  57,  69,  56,
					14,  17,  22,  29,  51,  87,  80,  62,
					18,  22,  37,  56,  68, 109, 103,  77,
					24,  35,  55,  64,  81, 104, 113,  92,
					49,  64,  78,  87, 103, 121, 120, 101,
					72,  92,  95,  98, 112, 100, 103,  99
				};
				/**
				* 量子化テーブル色差用
				* ファイル出力、内部処理兼用
				*/
				__device__ __constant__ static const int	component[] = {
					17,  18,  24,  47,  99,  99,  99,  99,
					18,  21,  26,  66,  99,  99,  99,  99,
					24,  26,  56,  99,  99,  99,  99,  99,
					47,  66,  99,  99,  99,  99,  99,  99,
					99,  99,  99,  99,  99,  99,  99,  99,
					99,  99,  99,  99,  99,  99,  99,  99,
					99,  99,  99,  99,  99,  99,  99,  99,
					99,  99,  99,  99,  99,  99,  99,  99
				};
			}  // namespace Quantize

			//----------------------------------------------------------------------------
			// ハフマン符号化用
			//============================================================================
			namespace HuffmanEncode {
				namespace DC {
					namespace luminance {
						/** 輝度DC成分用サイズテーブル */
						__device__ __constant__ static const u_int	code_size[] = {
							0x0002, 0x0003, 0x0003, 0x0003,
							0x0003, 0x0003, 0x0004, 0x0005,
							0x0006, 0x0007, 0x0008, 0x0009
						};

						/** 輝度DC成分用符号語テーブル */
						__device__ __constant__ static const u_int	code[] = {
							0x0000, 0x0002, 0x0003, 0x0004,
							0x0005, 0x0006, 0x000e, 0x001e,
							0x003e, 0x007e, 0x00fe, 0x01fe
						};
					}  // namespace luminance

					namespace component {
						/** 色差DC成分用サイズテーブル */
						__device__ __constant__ static const u_int	code_size[] = {
							0x0002, 0x0002, 0x0002, 0x0003,
							0x0004, 0x0005, 0x0006, 0x0007,
							0x0008, 0x0009, 0x000a, 0x000b
						};

						/** 色差DC成分用符号語テーブル */
						__device__ __constant__ static const u_int	code[] = {
							0x0000, 0x0001, 0x0002, 0x0006,
							0x000e, 0x001e, 0x003e, 0x007e,
							0x00fe, 0x01fe, 0x03fe, 0x07fe
						};
					}  // namespace component
				}  // namespace DC

				namespace AC {
					namespace luminance {
						/** 輝度AC成分用サイズテーブル */
						__device__ __constant__ static const u_int	code_size[] = {
							 4, // EOB
							 2,  2,  3,  4,  5,  7,  8,	10, 16, 16,
							 4,  5,  7,  9, 11,	16, 16, 16, 16, 16,
							 5,  8, 10,	12, 16, 16, 16, 16, 16, 16,
							 6,	 9, 12, 16, 16, 16, 16, 16, 16,	16,
							 6, 10, 16, 16, 16, 16, 16,	16, 16, 16,
							 7, 11, 16, 16, 16,	16, 16, 16, 16, 16,
							 7, 12, 16,	16, 16, 16, 16, 16, 16, 16,
							 8,	12, 16, 16, 16, 16, 16, 16, 16,	16,
							 9, 15, 16, 16, 16, 16, 16,	16, 16, 16,
							 9, 16, 16, 16, 16,	16, 16, 16, 16, 16,
							 9, 16, 16,	16, 16, 16, 16, 16, 16, 16,
							10,	16, 16, 16, 16, 16, 16, 16, 16,	16,
							10, 16, 16, 16, 16, 16, 16,	16, 16, 16,
							11, 16, 16, 16, 16,	16, 16, 16, 16, 16,
							16, 16, 16,	16, 16, 16, 16, 16, 16, 16,
							11,	// ZRL
							16, 16, 16, 16, 16, 16, 16, 16,	16, 16
						};

						/** 輝度AC成分用符号語テーブル */
						__device__ __constant__ static const u_int	code[] = {
							0x000a, // EOB
							0x0000, 0x0001, 0x0004,	0x000b, 0x001a, 0x0078, 0x00f8,	0x03f6, 0xff82, 0xff83,// ZRL
							0x000c,	0x001b, 0x0079, 0x01f6, 0x07f6,	0xff84, 0xff85, 0xff86, 0xff87,	0xff88,
							0x001c, 0x00f9, 0x03f7,	0x0ff4, 0xff89, 0xff8a, 0xff8b,	0xff8c, 0xff8d, 0xff8e,
							0x003a,	0x01f7, 0x0ff5, 0xff8f, 0xff90,	0xff91, 0xff92, 0xff93, 0xff94,	0xff95,
							0x003b, 0x03f8, 0xff96,	0xff97, 0xff98, 0xff99, 0xff9a,	0xff9b, 0xff9c, 0xff9d,
							0x007a,	0x07f7, 0xff9e, 0xff9f, 0xffa0,	0xffa1, 0xffa2, 0xffa3, 0xffa4,	0xffa5,
							0x007b, 0x0ff6, 0xffa6,	0xffa7, 0xffa8, 0xffa9, 0xffaa,	0xffab, 0xffac, 0xffad,
							0x00fa,	0x0ff7, 0xffae, 0xffaf, 0xffb0,	0xffb1, 0xffb2, 0xffb3, 0xffb4,	0xffb5,
							0x01f8, 0x7fc0, 0xffb6,	0xffb7, 0xffb8, 0xffb9, 0xffba,	0xffbb, 0xffbc, 0xffbd,
							0x01f9,	0xffbe, 0xffbf, 0xffc0, 0xffc1,	0xffc2, 0xffc3, 0xffc4, 0xffc5,	0xffc6,
							0x01fa, 0xffc7, 0xffc8,	0xffc9, 0xffca, 0xffcb, 0xffcc,	0xffcd, 0xffce, 0xffcf,
							0x03f9,	0xffd0, 0xffd1, 0xffd2, 0xffd3,	0xffd4, 0xffd5, 0xffd6, 0xffd7,	0xffd8,
							0x03fa, 0xffd9, 0xffda,	0xffdb, 0xffdc, 0xffdd, 0xffde,	0xffdf, 0xffe0, 0xffe1,
							0x07f8,	0xffe2, 0xffe3, 0xffe4, 0xffe5,	0xffe6, 0xffe7, 0xffe8, 0xffe9,	0xffea,
							0xffeb, 0xffec, 0xffed,	0xffee, 0xffef, 0xfff0, 0xfff1,	0xfff2, 0xfff3, 0xfff4,
							0x07f9,	// ZRL
							0xfff5, 0xfff6, 0xfff7, 0xfff8,	0xfff9, 0xfffa, 0xfffb, 0xfffc,	0xfffd, 0xfffe
						};
						__device__ __constant__ static const u_int	EOB = 0;	//! EOBのインデックス
						__device__ __constant__ static const u_int	ZRL = 151;//! ZRLのインデックス
					}  // namespace luminance
					namespace component {
						/** 色差AC成分用サイズテーブル */
						__device__ __constant__ static const u_int	code_size[] = {
							 2, // EOB
							 2,  3,  4,  5,  5,  6,  7,	 9, 10, 12,
							 4,  6,  8,  9, 11,	12, 16, 16, 16, 16,
							 5,  8, 10, 12, 15, 16, 16, 16, 16, 16,
							 5,	 8, 10, 12, 16, 16, 16, 16, 16,	16,
							 6,  9, 16, 16, 16, 16, 16,	16, 16, 16,
							 6, 10, 16, 16, 16,	16, 16, 16, 16, 16,
							 7, 11, 16,	16, 16, 16, 16, 16, 16, 16,
							 7,	11, 16, 16, 16, 16, 16, 16, 16,	16,
							 8, 16, 16, 16, 16, 16, 16,	16, 16, 16,
							 9, 16, 16, 16, 16,	16, 16, 16, 16, 16,
							 9, 16, 16,	16, 16, 16, 16, 16, 16, 16,
							 9,	16, 16, 16, 16, 16, 16, 16, 16,	16,
							 9, 16, 16, 16, 16, 16, 16,	16, 16, 16,
							11, 16, 16, 16, 16, 16, 16, 16, 16, 16,
							14, 16, 16, 16, 16, 16, 16, 16, 16, 16,
							10, // ZRL
							15, 16, 16, 16, 16, 16, 16, 16,	16, 16
						};

						/** 色差AC成分用符号語テーブル */
						__device__ __constant__ static const u_int	code[] = {
							0x0000, // EOB
							0x0001, 0x0004, 0x000a,	0x0018, 0x0019, 0x0038, 0x0078,	0x01f4, 0x03f6, 0x0ff4,
							0x000b,	0x0039, 0x00f6, 0x01f5, 0x07f6,	0x0ff5, 0xff88, 0xff89, 0xff8a,	0xff8b,
							0x001a, 0x00f7, 0x03f7,	0x0ff6, 0x7fc2, 0xff8c, 0xff8d,	0xff8e, 0xff8f, 0xff90,
							0x001b,	0x00f8, 0x03f8, 0x0ff7, 0xff91,	0xff92, 0xff93, 0xff94, 0xff95, 0xff96,
							0x003a, 0x01f6, 0xff97,	0xff98, 0xff99, 0xff9a, 0xff9b,	0xff9c, 0xff9d, 0xff9e,
							0x003b,	0x03f9, 0xff9f, 0xffa0, 0xffa1,	0xffa2, 0xffa3, 0xffa4, 0xffa5,	0xffa6,
							0x0079, 0x07f7, 0xffa7,	0xffa8, 0xffa9, 0xffaa, 0xffab,	0xffac, 0xffad, 0xffae,
							0x007a,	0x07f8, 0xffaf, 0xffb0, 0xffb1,	0xffb2, 0xffb3, 0xffb4, 0xffb5,	0xffb6,
							0x00f9, 0xffb7, 0xffb8,	0xffb9, 0xffba, 0xffbb, 0xffbc,	0xffbd, 0xffbe, 0xffbf,
							0x01f7,	0xffc0, 0xffc1, 0xffc2, 0xffc3,	0xffc4, 0xffc5, 0xffc6, 0xffc7,	0xffc8,
							0x01f8, 0xffc9, 0xffca,	0xffcb, 0xffcc, 0xffcd, 0xffce,	0xffcf, 0xffd0, 0xffd1,
							0x01f9,	0xffd2, 0xffd3, 0xffd4, 0xffd5,	0xffd6, 0xffd7, 0xffd8, 0xffd9,	0xffda,
							0x01fa, 0xffdb, 0xffdc,	0xffdd, 0xffde, 0xffdf, 0xffe0,	0xffe1, 0xffe2, 0xffe3,
							0x07f9,	0xffe4, 0xffe5, 0xffe6, 0xffe7,	0xffe8, 0xffe9, 0xffea, 0xffeb,	0xffec,
							0x3fe0, 0xffed, 0xffee,	0xffef, 0xfff0, 0xfff1, 0xfff2,	0xfff3, 0xfff4, 0xfff5,
							0x03fa,	// ZRL
							0x7fc3, 0xfff6, 0xfff7, 0xfff8,	0xfff9, 0xfffa, 0xfffb, 0xfffc,	0xfffd, 0xfffe
						};

						__device__ __constant__ static const u_int		EOB = 0;	//! EOBのインデックス
						__device__ __constant__ static const u_int		ZRL = 151;	//! ZRLのインデックス
					}  // namespace component
				}  // namespace AC
			}  // namespace HuffmanEncode

//			namespace HuffmanDecode {
//				/** @brief テーブルセット */
//				struct TableSet {
//					const u_int table_size; //! テーブル要素数
//					const u_int* size_table; //! ハフマンサイズテーブル
//					const u_int* code_table; //! ハフマン符号語テーブル
//					const u_int* value_table; //! ハフマンパラメータテーブル
//
//				public:
//					TableSet(u_int s, const u_int *st, const u_int *ct, const u_int *vt) :
//						table_size(s),
//						size_table(st),
//						code_table(ct),
//						value_table(vt) {
//					}
//				};
//
//				namespace DC {
//					/** @brief ハフマンテーブル輝度 */
//					namespace luminance {
//						/** 輝度DC成分用サイズテーブル */
//						__device__ __constant__ static const u_int size[] = {
//							0x0002, 0x0003, 0x0003, 0x0003,
//							0x0003, 0x0003, 0x0004, 0x0005,
//							0x0006, 0x0007, 0x0008, 0x0009
//						};
//						/** 輝度DC成分用符号語テーブル */
//						__device__ __constant__ static const u_int code[] = {
//							0x0000, 0x0002, 0x0003, 0x0004,
//							0x0005, 0x0006, 0x000e, 0x001e,
//							0x003e, 0x007e, 0x00fe, 0x01fe
//						};
//						/** 輝度DC成分用パラメータ */
//						__device__ __constant__ static const u_int param[] = {
//							0x0000, 0x0001, 0x0002, 0x0003,
//							0x0004, 0x0005, 0x0006, 0x0007,
//							0x0008, 0x0009, 0x000a, 0x000b
//						};
//					};
//					/** @brief ハフマンテーブル色差 */
//					namespace component {
//						/** 色差DC成分用サイズテーブル */
//						__device__ __constant__ static const u_int	size[] = {
//							0x0002, 0x0002, 0x0002, 0x0003,
//							0x0004, 0x0005, 0x0006, 0x0007,
//							0x0008, 0x0009, 0x000a, 0x000b
//						};
//						/** 色差DC成分用符号語テーブル */
//						__device__ __constant__ static const u_int	code[] = {
//							0x0000, 0x0001, 0x0002, 0x0006,
//							0x000e, 0x001e, 0x003e, 0x007e,
//							0x00fe, 0x01fe, 0x03fe, 0x07fe
//						};
//						/** 色差DC成分用パラメータ */
//						__device__ __constant__ static const u_int	param[] = {
//							0x0000, 0x0001, 0x0002, 0x0003,
//							0x0004, 0x0005, 0x0006, 0x0007,
//							0x0008, 0x0009, 0x000a, 0x000b
//						};
//					};
//				}
//
//				namespace AC {
//					/** @brief ハフマンテーブル輝度 */
//					namespace luminance {
//						/** 輝度AC成分用サイズテーブル */
//						__device__ __constant__ static const u_int	size[] = {
//							 2,  2,  3,  4,  4,  4,  5,  5,
//							 5,  6,  6,  7,  7,  7,  7,  8,
//							 8,  8,  9,  9,  9,  9,  9, 10,
//							10, 10, 10, 10, 11, 11, 11, 11,
//							12, 12, 12, 12, 15, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16
//						};
//
//						/** 輝度AC成分用符号語テーブル */
//						__device__ __constant__ static const u_int	code[] = {
//							0x0000, 0x0001, 0x0004, 0x000a, 0x000b, 0x000c, 0x001a, 0x001b,
//							0x001c, 0x003a, 0x003b, 0x0078, 0x0079, 0x007a, 0x007b, 0x00f8,
//							0x00f9, 0x00fa, 0x01f6, 0x01f7, 0x01f8, 0x01f9, 0x01fa, 0x03f6,
//							0x03f7, 0x03f8, 0x03f9, 0x03fa, 0x07f6, 0x07f7, 0x07f8, 0x07f9,
//							0x0ff4, 0x0ff5, 0x0ff6, 0x0ff7, 0x7fc0, 0xff82, 0xff83, 0xff84,
//							0xff85, 0xff86, 0xff87, 0xff88, 0xff89, 0xff8a, 0xff8b, 0xff8c,
//							0xff8d, 0xff8e, 0xff8f, 0xff90, 0xff91, 0xff92, 0xff93, 0xff94,
//							0xff95, 0xff96, 0xff97, 0xff98, 0xff99, 0xff9a, 0xff9b, 0xff9c,
//							0xff9d, 0xff9e, 0xff9f, 0xffa0, 0xffa1, 0xffa2, 0xffa3, 0xffa4,
//							0xffa5, 0xffa6, 0xffa7, 0xffa8, 0xffa9, 0xffaa, 0xffab, 0xffac,
//							0xffad, 0xffae, 0xffaf, 0xffb0, 0xffb1, 0xffb2, 0xffb3, 0xffb4,
//							0xffb5, 0xffb6, 0xffb7, 0xffb8, 0xffb9, 0xffba, 0xffbb, 0xffbc,
//							0xffbd, 0xffbe, 0xffbf, 0xffc0, 0xffc1, 0xffc2, 0xffc3, 0xffc4,
//							0xffc5, 0xffc6, 0xffc7, 0xffc8, 0xffc9, 0xffca, 0xffcb, 0xffcc,
//							0xffcd, 0xffce, 0xffcf, 0xffd0, 0xffd1, 0xffd2, 0xffd3, 0xffd4,
//							0xffd5, 0xffd6, 0xffd7, 0xffd8, 0xffd9, 0xffda, 0xffdb, 0xffdc,
//							0xffdd, 0xffde, 0xffdf, 0xffe0, 0xffe1, 0xffe2, 0xffe3, 0xffe4,
//							0xffe5, 0xffe6, 0xffe7, 0xffe8, 0xffe9, 0xffea, 0xffeb, 0xffec,
//							0xffed, 0xffee, 0xffef, 0xfff0, 0xfff1, 0xfff2, 0xfff3, 0xfff4,
//							0xfff5, 0xfff6, 0xfff7, 0xfff8, 0xfff9, 0xfffa, 0xfffb, 0xfffc,
//							0xfffd, 0xfffe,
//						};
//
//						/** 輝度AC成分用パラメータ */
//						__device__ __constant__ static const u_int param[] = {
//							0x0001, 0x0002, 0x0003, 0x0000, 0x0004, 0x0011, 0x0005, 0x0012,
//							0x0021, 0x0031, 0x0041, 0x0006, 0x0013, 0x0051, 0x0061, 0x0007,
//							0x0022, 0x0071, 0x0014, 0x0032, 0x0081, 0x0091, 0x00a1, 0x0008,
//							0x0023, 0x0042, 0x00b1, 0x00c1, 0x0015, 0x0052, 0x00d1, 0x00f0,
//							0x0024, 0x0033, 0x0062, 0x0072, 0x0082, 0x0009, 0x000a, 0x0016,
//							0x0017, 0x0018, 0x0019, 0x001a, 0x0025, 0x0026, 0x0027, 0x0028,
//							0x0029, 0x002a, 0x0034, 0x0035, 0x0036, 0x0037, 0x0038, 0x0039,
//							0x003a, 0x0043, 0x0044, 0x0045, 0x0046, 0x0047, 0x0048, 0x0049,
//							0x004a, 0x0053, 0x0054, 0x0055, 0x0056, 0x0057, 0x0058, 0x0059,
//							0x005a, 0x0063, 0x0064, 0x0065, 0x0066, 0x0067, 0x0068, 0x0069,
//							0x006a, 0x0073, 0x0074, 0x0075, 0x0076, 0x0077, 0x0078, 0x0079,
//							0x007a, 0x0083, 0x0084, 0x0085, 0x0086, 0x0087, 0x0088, 0x0089,
//							0x008a, 0x0092, 0x0093, 0x0094, 0x0095, 0x0096, 0x0097, 0x0098,
//							0x0099, 0x009a, 0x00a2, 0x00a3, 0x00a4, 0x00a5, 0x00a6, 0x00a7,
//							0x00a8, 0x00a9, 0x00aa, 0x00b2, 0x00b3, 0x00b4, 0x00b5, 0x00b6,
//							0x00b7, 0x00b8, 0x00b9, 0x00ba, 0x00c2, 0x00c3, 0x00c4, 0x00c5,
//							0x00c6, 0x00c7, 0x00c8, 0x00c9, 0x00ca, 0x00d2, 0x00d3, 0x00d4,
//							0x00d5, 0x00d6, 0x00d7, 0x00d8, 0x00d9, 0x00da, 0x00e1, 0x00e2,
//							0x00e3, 0x00e4, 0x00e5, 0x00e6, 0x00e7, 0x00e8, 0x00e9, 0x00ea,
//							0x00f1, 0x00f2, 0x00f3, 0x00f4, 0x00f5, 0x00f6, 0x00f7, 0x00f8,
//							0x00f9, 0x00fa
//						};
//
//						static const u_int EOB = 0;
//						static const u_int ZRL = 151;
//					};
//					/** @brief ハフマンテーブル色差 */
//					namespace component {
//						/** 色差AC成分用サイズテーブル */
//						__device__ __constant__ static const u_int	size[] = {
//							 2,  2,  3,  4,  4,  5,  5,  5,
//							 5,  6,  6,  6,  6,  7,  7,  7,
//							 8,  8,  8,  8,  9,  9,  9,  9,
//							 9,  9,  9, 10, 10, 10, 10, 10,
//							11, 11, 11, 11, 12, 12, 12, 12,
//							14, 15, 15, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16, 16, 16, 16, 16, 16, 16,
//							16, 16,
//						};
//
//						/** 色差AC成分用符号語テーブル */
//						__device__ __constant__ static const u_int	code[] = {
//							0x0000, 0x0001, 0x0004, 0x000a, 0x000b, 0x0018, 0x0019, 0x001a,
//							0x001b, 0x0038, 0x0039, 0x003a, 0x003b, 0x0078, 0x0079, 0x007a,
//							0x00f6, 0x00f7, 0x00f8, 0x00f9, 0x01f4, 0x01f5, 0x01f6, 0x01f7,
//							0x01f8, 0x01f9, 0x01fa, 0x03f6, 0x03f7, 0x03f8, 0x03f9, 0x03fa,
//							0x07f6, 0x07f7, 0x07f8, 0x07f9, 0x0ff4, 0x0ff5, 0x0ff6, 0x0ff7,
//							0x3fe0, 0x7fc2, 0x7fc3, 0xff88, 0xff89, 0xff8a, 0xff8b, 0xff8c,
//							0xff8d, 0xff8e, 0xff8f, 0xff90, 0xff91, 0xff92, 0xff93, 0xff94,
//							0xff95, 0xff96, 0xff97, 0xff98, 0xff99, 0xff9a, 0xff9b, 0xff9c,
//							0xff9d, 0xff9e, 0xff9f, 0xffa0, 0xffa1, 0xffa2, 0xffa3, 0xffa4,
//							0xffa5, 0xffa6, 0xffa7, 0xffa8, 0xffa9, 0xffaa, 0xffab, 0xffac,
//							0xffad, 0xffae, 0xffaf, 0xffb0, 0xffb1, 0xffb2, 0xffb3, 0xffb4,
//							0xffb5, 0xffb6, 0xffb7, 0xffb8, 0xffb9, 0xffba, 0xffbb, 0xffbc,
//							0xffbd, 0xffbe, 0xffbf, 0xffc0, 0xffc1, 0xffc2, 0xffc3, 0xffc4,
//							0xffc5, 0xffc6, 0xffc7, 0xffc8, 0xffc9, 0xffca, 0xffcb, 0xffcc,
//							0xffcd, 0xffce, 0xffcf, 0xffd0, 0xffd1, 0xffd2, 0xffd3, 0xffd4,
//							0xffd5, 0xffd6, 0xffd7, 0xffd8, 0xffd9, 0xffda, 0xffdb, 0xffdc,
//							0xffdd, 0xffde, 0xffdf, 0xffe0, 0xffe1, 0xffe2, 0xffe3, 0xffe4,
//							0xffe5, 0xffe6, 0xffe7, 0xffe8, 0xffe9, 0xffea, 0xffeb, 0xffec,
//							0xffed, 0xffee, 0xffef, 0xfff0, 0xfff1, 0xfff2, 0xfff3, 0xfff4,
//							0xfff5, 0xfff6, 0xfff7, 0xfff8, 0xfff9, 0xfffa, 0xfffb, 0xfffc,
//							0xfffd, 0xfffe,
//						};
//
//						/** 色差AC成分用パラメータ */
//						__device__ __constant__ static const u_int param[]={//
//							0x0000, 0x0001, 0x0002, 0x0003, 0x0011, 0x0004, 0x0005, 0x0021,
//							0x0031, 0x0006, 0x0012, 0x0041, 0x0051, 0x0007, 0x0061, 0x0071,
//							0x0013, 0x0022, 0x0032, 0x0081, 0x0008, 0x0014, 0x0042, 0x0091,
//							0x00a1, 0x00b1, 0x00c1, 0x0009, 0x0023, 0x0033, 0x0052, 0x00f0,
//							0x0015, 0x0062, 0x0072, 0x00d1, 0x000a, 0x0016, 0x0024, 0x0034,
//							0x00e1, 0x0025, 0x00f1, 0x0017, 0x0018, 0x0019, 0x001a, 0x0026,
//							0x0027, 0x0028, 0x0029, 0x002a, 0x0035, 0x0036, 0x0037, 0x0038,
//							0x0039, 0x003a, 0x0043, 0x0044, 0x0045, 0x0046, 0x0047, 0x0048,
//							0x0049, 0x004a, 0x0053, 0x0054, 0x0055, 0x0056, 0x0057, 0x0058,
//							0x0059, 0x005a, 0x0063, 0x0064, 0x0065, 0x0066, 0x0067, 0x0068,
//							0x0069, 0x006a, 0x0073, 0x0074, 0x0075, 0x0076, 0x0077, 0x0078,
//							0x0079, 0x007a, 0x0082, 0x0083, 0x0084, 0x0085, 0x0086, 0x0087,
//							0x0088, 0x0089, 0x008a, 0x0092, 0x0093, 0x0094, 0x0095, 0x0096,
//							0x0097, 0x0098, 0x0099, 0x009a, 0x00a2, 0x00a3, 0x00a4, 0x00a5,
//							0x00a6, 0x00a7, 0x00a8, 0x00a9, 0x00aa, 0x00b2, 0x00b3, 0x00b4,
//							0x00b5, 0x00b6, 0x00b7, 0x00b8, 0x00b9, 0x00ba, 0x00c2, 0x00c3,
//							0x00c4, 0x00c5, 0x00c6, 0x00c7, 0x00c8, 0x00c9, 0x00ca, 0x00d2,
//							0x00d3, 0x00d4, 0x00d5, 0x00d6, 0x00d7, 0x00d8, 0x00d9, 0x00da,
//							0x00e2, 0x00e3, 0x00e4, 0x00e5, 0x00e6, 0x00e7, 0x00e8, 0x00e9,
//							0x00ea, 0x00f2, 0x00f3, 0x00f4, 0x00f5, 0x00f6, 0x00f7, 0x00f8,
//							0x00f9, 0x00fa
//						};
//
//						static const u_int EOB = 0;
//						static const u_int ZRL = 151;
//					};
//				}
//
//				__device__ __constant__ static const TableSet tableSet[][2] = {
//					{
//						TableSet(
//							12,
//							DC::luminance::size,
//							DC::luminance::code,
//							DC::luminance::param
//						),
//						TableSet(
//							12,
//							DC::component::size,
//							DC::component::code,
//							DC::component::param
//						)
//					},
//					{
//						TableSet(
//							162,
//							AC::luminance::size,
//							AC::luminance::code,
//							AC::luminance::param
//						),
//						TableSet(
//							162,
//							AC::component::size,
//							AC::component::code,
//							AC::component::param
//						)
//					}
//				};
//			}

			namespace Sampling {
				/**
				* 間引き用テーブル輝度用
				* ファイル出力、内部処理兼用
				*/
				static const u_int	luminance[] = {
					 0,  0,  1,  1,  2,  2,  3,  3,  8,  8,  9,  9, 10, 10, 11, 11,
					 0,  0,  1,  1,  2,  2,  3,  3,  8,  8,  9,  9, 10, 10, 11, 11,
					16, 16, 17, 17, 18, 18, 19, 19, 24, 24, 25, 25, 26, 26, 27, 27,
					16, 16, 17, 17, 18, 18, 19, 19, 24, 24, 25, 25, 26, 26, 27, 27,

					 4,  4,  5,  5,  6,  6,  7,  7, 12, 12, 13, 13, 14, 14, 15, 15,
					 4,  4,  5,  5,  6,  6,  7,  7, 12, 12, 13, 13, 14, 14, 15, 15,
					20, 20, 21, 21, 22, 22, 23, 23, 28, 28, 29, 29, 30, 30, 31, 31,
					20, 20, 21, 21, 22, 22, 23, 23, 28, 28, 29, 29, 30, 30, 31, 31,

					32, 32, 33, 33, 34, 34, 35, 35, 40, 40, 41, 41, 42, 42, 43, 43,
					32, 32, 33, 33, 34, 34, 35, 35, 40, 40, 41, 41, 42, 42, 43, 43,
					48, 48, 49, 49, 50, 50, 51, 51, 56, 56, 57, 57, 58, 58, 59, 59,
					48, 48, 49, 49, 50, 50, 51, 51, 56, 56, 57, 57, 58, 58, 59, 59,

					36, 36, 37, 37, 38, 38, 39, 39, 44, 44, 45, 45, 46, 46, 47, 47,
					36, 36, 37, 37, 38, 38, 39, 39, 44, 44, 45, 45, 46, 46, 47, 47,
					52, 52, 53, 53, 54, 54, 55, 55, 60, 60, 61, 61, 62, 62, 63, 63,
					52, 52, 53, 53, 54, 54, 55, 55, 60, 60, 61, 61, 62, 62, 63, 63
				};
			}  // namespace Sampling
		}  // namespace encode_table
	}  // namespace cuda
}  // namespace jpeg

#endif
