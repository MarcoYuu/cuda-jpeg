/*
 * debug_log.cpp
 *
 *  Created on: 2012/11/28
 *      Author: yuumomma
 */

#include "debug_log.h"

#include <iostream>
#include <fstream>
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>

#include "timer.h"
#include "cuda/cuda_timer.h"

namespace util {
	using namespace std;
	using namespace util;

	static boost::shared_ptr<WatchInterface> watch;
	static bool is_print = true;
	static bool is_export = true;

	enum Section {
		ROOT, SECTION, SUBSECTION
	} section_ = ROOT;

	void DebugLog::initTimer(bool cpu) {
		if (cpu) {
			watch.reset(new StopWatch());
		} else {
			watch.reset(new cuda::CudaStopWatch());
		}
	}

	void DebugLog::enablePrint(bool enable) {
		is_print = enable;
	}
	void DebugLog::enableExport(bool enable) {
		is_export = enable;
	}

	bool DebugLog::isPrint() {
		return is_print;
	}
	bool DebugLog::isExport() {
		return is_export;
	}

	void DebugLog::startSection(const string &section) {
		if (!is_print)
			return;
		cout << "=================================================\n";
		cout << " " << section << "\n";
		cout << "-------------------------------------------------\n" << endl;
		section_ = SECTION;
	}
	void DebugLog::endSection(const string &section) {
		if (!is_print)
			return;
		cout << "-------------------------------------------------\n";
		cout << " " << section << "\n";
		cout << "=================================================\n" << endl;
		section_ = ROOT;
	}
	void DebugLog::startSubSection(const string &section) {
		if (!is_print)
			return;
		cout << "	-----------------------------------------------\n";
		cout << "	 " << section << "\n";
		cout << "	-----------------------------------------------\n" << endl;
		section_ = SUBSECTION;
	}
	void DebugLog::endSubSection() {
		if (!is_print)
			return;
		cout << "\n	=================================================\n" << endl;
		section_ = SECTION;
	}
	void DebugLog::log(const string &comment) {
		if (!is_print)
			return;
		string space;
		switch (section_) {
		case ROOT:
			space = "";
			break;
		case SECTION:
			space = "	";
			break;
		case SUBSECTION:
			space = "		";
			break;
		default:
			space = "";
			break;
		}
		cout << space << comment << "\n";
	}

	void DebugLog::startLoggingTime(const string &tag) {
		if (!is_print)
			return;
		cout << "	 	" << tag << ": ";
		watch->start();
	}
	void DebugLog::endLoggingTime() {
		if (!is_print)
			return;
		watch->lap();
		watch->stop();
		cout << watch->getLastElapsedTime() * 1000 << "[ms]" << endl;
	}
	void DebugLog::printTotalTime() {
		if (!is_print)
			return;
		string total = boost::lexical_cast<string>(watch->getTotalTime() * 1000);
		log("Total Time: " + total + "[ms]");
	}
	void DebugLog::resetTotalTime() {
		if (!is_print)
			return;
		watch->clear();
	}

	void DebugLog::dump_memory(void* data, size_t data_size, const string &filename) {
		if (!isExport())
			return;
		ofstream ofs(filename.c_str(), ios::binary);
		ofs.write((char*) data, data_size);
	}
}

