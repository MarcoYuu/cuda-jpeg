/*
 * encoder_table.cpp
 *
 *  Created on: 2012/10/28
 *      Author: yuumomma
 */

#include "encoder_tables.h"

namespace encode_table{

	/** ジグザグシーケンス用  */
	const int	Zigzag::sequence[] = {
		 0,  1,  5,  6, 14, 15, 27, 28,
		 2,  4,  7, 13, 16, 26, 29, 42,
		 3,  8, 12, 17, 25, 30, 41, 43,
		 9, 11, 18, 24, 31, 40, 44, 53,
		10, 19, 23, 32, 39, 45, 52, 54,
		20, 22, 33, 38, 46, 51, 55, 60,
		21, 34, 37, 47, 50, 56, 59, 61,
		35, 36, 48, 49, 57, 58, 62, 63
	};

	/** 量子化テーブル輝度用 */
	const int	Quantize::luminance[] = {
		16,  11,  10,  16,  24,  40,  51,  61,
		12,  12,  14,  19,  26,  58,  60,  55,
		14,  13,  16,  24,  40,  57,  69,  56,
		14,  17,  22,  29,  51,  87,  80,  62,
		18,  22,  37,  56,  68, 109, 103,  77,
		24,  35,  55,  64,  81, 104, 113,  92,
		49,  64,  78,  87, 103, 121, 120, 101,
		72,  92,  95,  98, 112, 100, 103,  99
	};

	/** 量子化テーブル色差用 */
	const int	Quantize::component[] = {
		17, 18, 24, 47, 99, 99, 99, 99,
		18, 21, 26, 66, 99, 99, 99, 99,
		24, 26, 56, 99, 99, 99, 99, 99,
		47, 66, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99
	};

	/** 輝度DC成分用サイズテーブル */
	const int	HuffmanEncode::DC::luminance::size[] = {
		0x0002, 0x0003, 0x0003, 0x0003,
		0x0003, 0x0003, 0x0004, 0x0005,
		0x0006, 0x0007, 0x0008, 0x0009
	};

	/** 輝度DC成分用符号語テーブル */
	const int HuffmanEncode::DC::luminance::code[] = {
		0x0000, 0x0002, 0x0003, 0x0004,
		0x0005, 0x0006, 0x000e, 0x001e,
		0x003e, 0x007e, 0x00fe, 0x01fe
	};

	/** 色差DC成分用サイズテーブル */
	const int	HuffmanEncode::DC::component::size[] = {
		0x0002, 0x0002, 0x0002, 0x0003,
		0x0004, 0x0005, 0x0006, 0x0007,
		0x0008, 0x0009, 0x000a, 0x000b
	};

	/** 色差DC成分用符号語テーブル */
	const int	HuffmanEncode::DC::component::code[] = {
		0x0000, 0x0001, 0x0002, 0x0006,
		0x000e, 0x001e, 0x003e, 0x007e,
		0x00fe, 0x01fe, 0x03fe, 0x07fe
	};

	/** 輝度AC成分用サイズテーブル */
	const int	HuffmanEncode::AC::luminance::size[] = {
		 4,  2,  2,  3,  4,  5,  7,  8,
		10, 16, 16,  4,  5,  7,  9, 11,
		16, 16, 16, 16, 16,  5,  8, 10,
		12, 16, 16, 16, 16, 16, 16,  6,
		 9, 12, 16, 16, 16, 16, 16, 16,
		16,  6, 10, 16, 16, 16, 16, 16,
		16, 16, 16,  7, 11, 16, 16, 16,
		16, 16, 16, 16, 16,  7, 12, 16,
		16, 16, 16, 16, 16, 16, 16,  8,
		12, 16, 16, 16, 16, 16, 16, 16,
		16,  9, 15, 16, 16, 16, 16, 16,
		16, 16, 16,  9, 16, 16, 16, 16,
		16, 16, 16, 16, 16,  9, 16, 16,
		16, 16, 16, 16, 16, 16, 16, 10,
		16, 16, 16, 16, 16, 16, 16, 16,
		16, 10, 16, 16, 16, 16, 16, 16,
		16, 16, 16, 11, 16, 16, 16, 16,
		16, 16, 16, 16, 16, 16, 16, 16,
		16, 16, 16, 16, 16, 16, 16, 11,
		16, 16, 16, 16, 16, 16, 16, 16,
		16, 16
	};

	/** 輝度AC成分用符号語テーブル */
	const int	HuffmanEncode::AC::luminance::code[] = {
		0x000a, 0x0000, 0x0001, 0x0004,
		0x000b, 0x001a, 0x0078, 0x00f8,
		0x03f6, 0xff82, 0xff83, 0x000c,
		0x001b, 0x0079, 0x01f6, 0x07f6,
		0xff84, 0xff85, 0xff86, 0xff87,
		0xff88, 0x001c, 0x00f9, 0x03f7,
		0x0ff4, 0xff89, 0xff8a, 0xff8b,
		0xff8c, 0xff8d, 0xff8e, 0x003a,
		0x01f7, 0x0ff5, 0xff8f, 0xff90,
		0xff91, 0xff92, 0xff93, 0xff94,
		0xff95, 0x003b, 0x03f8, 0xff96,
		0xff97, 0xff98, 0xff99, 0xff9a,
		0xff9b, 0xff9c, 0xff9d, 0x007a,
		0x07f7, 0xff9e, 0xff9f, 0xffa0,
		0xffa1, 0xffa2, 0xffa3, 0xffa4,
		0xffa5, 0x007b, 0x0ff6, 0xffa6,
		0xffa7, 0xffa8, 0xffa9, 0xffaa,
		0xffab, 0xffac, 0xffad, 0x00fa,
		0x0ff7, 0xffae, 0xffaf, 0xffb0,
		0xffb1, 0xffb2, 0xffb3, 0xffb4,
		0xffb5, 0x01f8, 0x7fc0, 0xffb6,
		0xffb7, 0xffb8, 0xffb9, 0xffba,
		0xffbb, 0xffbc, 0xffbd, 0x01f9,
		0xffbe, 0xffbf, 0xffc0, 0xffc1,
		0xffc2, 0xffc3, 0xffc4, 0xffc5,
		0xffc6, 0x01fa, 0xffc7, 0xffc8,
		0xffc9, 0xffca, 0xffcb, 0xffcc,
		0xffcd, 0xffce, 0xffcf, 0x03f9,
		0xffd0, 0xffd1, 0xffd2, 0xffd3,
		0xffd4, 0xffd5, 0xffd6, 0xffd7,
		0xffd8, 0x03fa, 0xffd9, 0xffda,
		0xffdb, 0xffdc, 0xffdd, 0xffde,
		0xffdf, 0xffe0, 0xffe1, 0x07f8,
		0xffe2, 0xffe3, 0xffe4, 0xffe5,
		0xffe6, 0xffe7, 0xffe8, 0xffe9,
		0xffea, 0xffeb, 0xffec, 0xffed,
		0xffee, 0xffef, 0xfff0, 0xfff1,
		0xfff2, 0xfff3, 0xfff4, 0x07f9,
		0xfff5, 0xfff6, 0xfff7, 0xfff8,
		0xfff9, 0xfffa, 0xfffb, 0xfffc,
		0xfffd, 0xfffe
	};

	/** 色差AC成分用サイズテーブル */
	const int	HuffmanEncode::AC::component::size[] = {
		 2,  2,  3,  4,  5,  5,  6,  7,
		 9, 10, 12,  4,  6,  8,  9, 11,
		12, 16, 16, 16, 16,  5,  8, 10,
		12, 15, 16, 16, 16, 16, 16,  5,
		 8, 10, 12, 16, 16, 16, 16, 16,
		16,  6,  9, 16, 16, 16, 16, 16,
		16, 16, 16,  6, 10, 16, 16, 16,
		16, 16, 16, 16, 16,  7, 11, 16,
		16, 16, 16, 16, 16, 16, 16,  7,
		11, 16, 16, 16, 16, 16, 16, 16,
		16,  8, 16, 16, 16, 16, 16, 16,
		16, 16, 16,  9, 16, 16, 16, 16,
		16, 16, 16, 16, 16,  9, 16, 16,
		16, 16, 16, 16, 16, 16, 16,  9,
		16, 16, 16, 16, 16, 16, 16, 16,
		16,  9, 16, 16, 16, 16, 16, 16,
		16, 16, 16, 11, 16, 16, 16, 16,
		16, 16, 16, 16, 16, 14, 16, 16,
		16, 16, 16, 16, 16, 16, 16, 10,
		15, 16, 16, 16, 16, 16, 16, 16,
		16, 16
	};

	/** 色差AC成分用符号語テーブル */
	const int	HuffmanEncode::AC::component::code[] = {
		0x0000, 0x0001, 0x0004, 0x000a,
		0x0018, 0x0019, 0x0038, 0x0078,
		0x01f4, 0x03f6, 0x0ff4, 0x000b,
		0x0039, 0x00f6, 0x01f5, 0x07f6,
		0x0ff5, 0xff88, 0xff89, 0xff8a,
		0xff8b, 0x001a, 0x00f7, 0x03f7,
		0x0ff6, 0x7fc2, 0xff8c, 0xff8d,
		0xff8e, 0xff8f, 0xff90, 0x001b,
		0x00f8, 0x03f8, 0x0ff7, 0xff91,
		0xff92, 0xff93, 0xff94, 0xff95,
		0xff96, 0x003a, 0x01f6, 0xff97,
		0xff98, 0xff99, 0xff9a, 0xff9b,
		0xff9c, 0xff9d, 0xff9e, 0x003b,
		0x03f9, 0xff9f, 0xffa0, 0xffa1,
		0xffa2, 0xffa3, 0xffa4, 0xffa5,
		0xffa6, 0x0079, 0x07f7, 0xffa7,
		0xffa8, 0xffa9, 0xffaa, 0xffab,
		0xffac, 0xffad, 0xffae, 0x007a,
		0x07f8, 0xffaf, 0xffb0, 0xffb1,
		0xffb2, 0xffb3, 0xffb4, 0xffb5,
		0xffb6, 0x00f9, 0xffb7, 0xffb8,
		0xffb9, 0xffba, 0xffbb, 0xffbc,
		0xffbd, 0xffbe, 0xffbf, 0x01f7,
		0xffc0, 0xffc1, 0xffc2, 0xffc3,
		0xffc4, 0xffc5, 0xffc6, 0xffc7,
		0xffc8, 0x01f8, 0xffc9, 0xffca,
		0xffcb, 0xffcc, 0xffcd, 0xffce,
		0xffcf, 0xffd0, 0xffd1, 0x01f9,
		0xffd2, 0xffd3, 0xffd4, 0xffd5,
		0xffd6, 0xffd7, 0xffd8, 0xffd9,
		0xffda, 0x01fa, 0xffdb, 0xffdc,
		0xffdd, 0xffde, 0xffdf, 0xffe0,
		0xffe1, 0xffe2, 0xffe3, 0x07f9,
		0xffe4, 0xffe5, 0xffe6, 0xffe7,
		0xffe8, 0xffe9, 0xffea, 0xffeb,
		0xffec, 0x3fe0, 0xffed, 0xffee,
		0xffef, 0xfff0, 0xfff1, 0xfff2,
		0xfff3, 0xfff4, 0xfff5, 0x03fa,
		0x7fc3, 0xfff6, 0xfff7, 0xfff8,
		0xfff9, 0xfffa, 0xfffb, 0xfffc,
		0xfffd, 0xfffe
	};
}
