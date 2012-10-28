/**
* EncoderTables Ver.1.1.0 JPEG符号化クラス用テーブル
* Copyright 2004 AsWe.Co.Ltd. All rights reserved.
* The comments are written in Japanese (Shift-JIS).
*/

#ifndef ENCODER_TABLE_H_
#define ENCODER_TABLE_H_

namespace jpeg {
	namespace encode_table{

		//----------------------------------------------------------------------------
		// ジグザグシーケンス
		//============================================================================
		struct Zigzag {
			/** ジグザグシーケンス用  */
			static const int sequence[64];
		};

		//----------------------------------------------------------------------------
		// 量子化テーブル
		//============================================================================
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
				struct luminance {
					/** 輝度DC成分用サイズテーブル */
					static const int size[12];
					/** 輝度DC成分用符号語テーブル */
					static const int code[12];
				};
				struct component {
					/** 色差DC成分用サイズテーブル */
					static const int size[12];
					/** 色差DC成分用符号語テーブル */
					static const int code[12];
				};
			};

			namespace AC {
				struct luminance {
					/** 輝度AC成分用サイズテーブル */
					static const int size[162];
					/** 輝度AC成分用符号語テーブル */
					static const int code[162];

					static const int EOB =0;
					static const int ZRL =151;
				};
				struct component {
					/** 色差AC成分用サイズテーブル */
					static const int size[162];
					/** 色差AC成分用符号語テーブル */
					static const int code[162];

					static const int EOB =0;
					static const int ZRL =151;
				};
			}
		}

		//----------------------------------------------------------------------------
		// ハフマン復号用
		//============================================================================
		/**
		 * ハフマンテーブルの提供
		 *
		 * @author yuumomma
		 * @version 1.0
		 */
		namespace HuffmanDecode {
			struct TableSet {
				const int table_size;		//! テーブル要素数
				const int* size_table;		//! ハフマンサイズテーブル
				const int* code_table;		//! ハフマン符号語テーブル
				const int* value_table;	//! ハフマンパラメータテーブル

				static const TableSet luminanceDC;
				static const TableSet componentDC;
				static const TableSet luminanceAC;
				static const TableSet componentAC;

			private:
				TableSet(int s, const int *st, const int *ct, const int *vt) :
					table_size(s),
					size_table(st),
					code_table(ct),
					value_table(vt) {
				}
			};

			namespace DC {
				struct luminance {
					/** 輝度DC成分用サイズテーブル */
					static const int size[12];
					/** 輝度DC成分用符号語テーブル */
					static const int code[12];
					/** 輝度DC成分用パラメータ */
					static const int param[12];
				};
				struct component {
					/** 色差DC成分用サイズテーブル */
					static const int size[12];
					/** 色差DC成分用符号語テーブル */
					static const int code[12];
					/** 色差DC成分用パラメータ */
					static const int param[12];
				};
			}

			namespace AC {
				struct luminance {
					/** 輝度AC成分用サイズテーブル */
					static const int size[162];
					/** 輝度AC成分用符号語テーブル */
					static const int code[162];
					/** 輝度AC成分用パラメータ */
					static const int param[162];

					static const int EOB = 0;
					static const int ZRL = 151;
				};
				struct component {
					/** 色差AC成分用サイズテーブル */
					static const int size[162];
					/** 色差AC成分用符号語テーブル */
					static const int code[162];
					/** 色差AC成分用パラメータ */
					static const int param[162];

					static const int EOB = 0;
					static const int ZRL = 151;
				};
			}
		}

		struct Sampling{
			/**
			* 輝度用間引きテーブル
			*
			* ファイル出力、内部処理兼用
			*/
			static const int luminance[256];
		};
	}
}  // namespace cpu
#endif 
