/*
 * DebugLog.h
 *
 *  Created on: 2012/11/28
 *      Author: yuumomma
 */

#ifndef DEBUGLOG_H_
#define DEBUGLOG_H_

#include <string>

namespace util {
	/**
	 * @brief 簡易ログ出力クラス
	 *
	 * 適当なフォーマットでログを出力する.
	 *
	 * @author yuumomma
	 * @version 1.0
	 */
	class DebugLog {
	public:
		/**
		 * @brief 初期化する
		 *
		 * これ以前は時間計測関連の動作は保証されない
		 *
		 * @param cpu 時間計測をCPU/GPUどちらにするか
		 */
		static void initTimer(bool cpu = true);

		/**
		 * @brief 出力有効化
		 *
		 * @param enable 有効/無効
		 */
		static void enablePrint(bool enable);
		/**
		 * @brief ファイル出力有効化
		 *
		 * @param enable 有効/無効
		 */
		static void enableExport(bool enable);

		/**
		 * @brief 出力が有効かどうか
		 * @return 出力が有効かどうか
		 */
		static bool isPrint();
		/**
		 * @brief ファイル出力が有効かどうか
		 *
		 * @return ファイル出力が有効かどうか
		 */
		static bool isExport();

		/**
		 * @brief セクション開始
		 *
		 * @param section セクション名
		 */
		static void startSection(const std::string &section);
		/**
		 * @brief セクション終了
		 *
		 * @param section セクション名
		 */
		static void endSection(const std::string &section);
		/**
		 * @brief サブセクション開始
		 *
		 * @param section サブセクション名
		 */
		static void startSubSection(const std::string &section);
		/**
		 * サブセクション終了
		 */
		static void endSubSection();
		/**
		 * @brief 文字列出力
		 *
		 * @param comment 出したい文字
		 */
		static void log(const std::string &comment);

		/**
		 * @brief 時間計測をはじめる
		 *
		 * @param tag 計測名
		 */
		static void startLoggingTime(const std::string &tag);
		/**
		 * @brief 時間計測を終わる
		 */
		static void endLoggingTime();
		/**
		 * @brief 総計時間を出力
		 */
		static void printTotalTime();
		/**
		 * @brief 総計時間をリセット
		 */
		static void resetTotalTime();

		/**
		 * @brief メモリダンプ
		 *
		 * @param data データ
		 * @param data_size 長さ
		 * @param filename 保存名
		 */
		static void dump_memory(void* data, size_t data_size, const std::string &filename);

		/**
		 * @brief ファイルにログを吐く
		 *
		 * @sa OutputFormat
		 * @sa enableExport(bool enable)
		 *
		 * @param file_name 出力ファイル名
		 * @param format 吐き出し内容を書いた関数など
		 */
		template<class OutputFormat>
		static void exportToFile(const std::string &file_name, const OutputFormat &format) {
			if (!isExport())
				return;
			std::ofstream ofs(file_name.c_str());
			format(ofs);
		}

		/**
		 * @brief 出力用関数オブジェクト
		 *
		 * 継承する必要は必ずしも無い.std::ofstream&を受け取る関数呼び出しできるものであれば良い.
		 *
		 * @author yuumomma
		 * @version 1.0
		 */
		struct OutputFormat {
			virtual ~OutputFormat() {
			}
			virtual void operator()(std::ofstream&) const =0;
		};

	private:
		DebugLog();
		DebugLog(DebugLog &);
		void operator=(DebugLog &);
	};
} // namespace util

#endif /* DEBUGLOG_H_ */
