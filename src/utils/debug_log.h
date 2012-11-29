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
	class DebugLog {
	public:
		static void init(bool cpu = true);

		static void enablePrint(bool enable);
		static void enableExport(bool enable);

		static bool isPrint();
		static bool isExport();

		static void startSection(const std::string &section);
		static void endSection(const std::string &section);
		static void startSubSection(const std::string &section);
		static void endSubSection();
		static void log(const std::string &comment);

		static void startLoggingTime(const std::string &tag);
		static void endLoggingTime();
		static void printTotalTime();
		static void resetTotalTime();

		static void dump_memory(void* data, size_t data_size, const std::string &filename);

		template<class OutputFormat>
		static void exportToFile(const std::string &file_name, const OutputFormat &format) {
			if (!isExport())
				return;
			std::ofstream ofs(file_name.c_str());
			format(ofs);
		}

		struct OutputFormat {
			virtual void operator()(std::ofstream&) const =0;
		};

	private:
		DebugLog(DebugLog &);
		void operator=(DebugLog &);
	};
} // namespace util

#endif /* DEBUGLOG_H_ */
