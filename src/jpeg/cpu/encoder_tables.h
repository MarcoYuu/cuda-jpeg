/**
 * EncoderTables Ver.1.1.0 JPEG符号化クラス用テーブル
 * Copyright 2004 AsWe.Co.Ltd. All rights reserved.
 * The comments are written in Japanese (Shift-JIS).
 */

#ifndef ENCODER_TABLE_H_
#define ENCODER_TABLE_H_

#include <utils/type_definitions.h>

namespace jpeg {
	namespace cpu {
		namespace encode_table {

			using util::u_int;

			//----------------------------------------------------------------------------
			// ジグザグシーケンス
			//============================================================================
			/** @brief ジグザグシーケンス用  */
			struct Zigzag {
				/** ジグザグシーケンス用  */
				static const u_int sequence[64];
			};

			//----------------------------------------------------------------------------
			// 量子化テーブル
			//============================================================================
			/** @brief 量子化テーブル */
			struct Quantize {
				/** 量子化テーブル輝度用 */
				static const int luminance[64];
				/** 量子化テーブル色差用 */
				static const int component[64];
			};

			//----------------------------------------------------------------------------
			// ハフマン符号化用
			//============================================================================
			namespace HuffmanEncode {
				namespace DC {
					/** @brief ハフマンテーブル輝度 */
					struct luminance {
						/** 輝度DC成分用サイズテーブル */
						static const u_int size[12];
						/** 輝度DC成分用符号語テーブル */
						static const u_int code[12];
					};
					/** @brief ハフマンテーブル色差 */
					struct component {
						/** 色差DC成分用サイズテーブル */
						static const u_int size[12];
						/** 色差DC成分用符号語テーブル */
						static const u_int code[12];
					};
				}
				;

				namespace AC {
					/** @brief ハフマンテーブル輝度 */
					struct luminance {
						/** 輝度AC成分用サイズテーブル */
						static const u_int size[162];
						/** 輝度AC成分用符号語テーブル */
						static const u_int code[162];

						static const u_int EOB = 0;
						static const u_int ZRL = 151;
					};
					/** @brief ハフマンテーブル色差 */
					struct component {
						/** 色差AC成分用サイズテーブル */
						static const u_int size[162];
						/** 色差AC成分用符号語テーブル */
						static const u_int code[162];

						static const u_int EOB = 0;
						static const u_int ZRL = 151;
					};
				}
			}

			//----------------------------------------------------------------------------
			// ハフマン復号用
			//============================================================================
			namespace HuffmanDecode {
				/** @brief テーブルセット */
				struct TableSet {
					const u_int table_size; //! テーブル要素数
					const u_int* size_table; //! ハフマンサイズテーブル
					const u_int* code_table; //! ハフマン符号語テーブル
					const u_int* value_table; //! ハフマンパラメータテーブル

					static const TableSet luminanceDC;
					static const TableSet componentDC;
					static const TableSet luminanceAC;
					static const TableSet componentAC;

				private:
					TableSet(u_int s, const u_int *st, const u_int *ct, const u_int *vt) :
						table_size(s),
						size_table(st),
						code_table(ct),
						value_table(vt) {
					}
				};

				namespace DC {
					/** @brief ハフマンテーブル輝度 */
					struct luminance {
						/** 輝度DC成分用サイズテーブル */
						static const u_int size[12];
						/** 輝度DC成分用符号語テーブル */
						static const u_int code[12];
						/** 輝度DC成分用パラメータ */
						static const u_int param[12];
					};
					/** @brief ハフマンテーブル色差 */
					struct component {
						/** 色差DC成分用サイズテーブル */
						static const u_int size[12];
						/** 色差DC成分用符号語テーブル */
						static const u_int code[12];
						/** 色差DC成分用パラメータ */
						static const u_int param[12];
					};
				}

				namespace AC {
					/** @brief ハフマンテーブル輝度 */
					struct luminance {
						/** 輝度AC成分用サイズテーブル */
						static const u_int size[162];
						/** 輝度AC成分用符号語テーブル */
						static const u_int code[162];
						/** 輝度AC成分用パラメータ */
						static const u_int param[162];

						static const u_int EOB = 0;
						static const u_int ZRL = 151;
					};
					/** @brief ハフマンテーブル色差 */
					struct component {
						/** 色差AC成分用サイズテーブル */
						static const u_int size[162];
						/** 色差AC成分用符号語テーブル */
						static const u_int code[162];
						/** 色差AC成分用パラメータ */
						static const u_int param[162];

						static const u_int EOB = 0;
						static const u_int ZRL = 151;
					};
				}
			}

			/**
			 * @brief 輝度用間引きテーブル
			 *
			 * ファイル出力、内部処理兼用
			 */
			struct Sampling {

				static const u_int luminance[256];
			};
		}
	} // namespace cpu
} // namespace jpeg
#endif 
