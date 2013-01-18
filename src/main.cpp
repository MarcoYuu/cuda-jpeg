/*
 * main.cpp
 *
 *  Created on: 2012/09/21
 *      Author: Yuu Momma
 */

#include <cstdlib>
#include <string>
#include <iostream>

#include <boost/lexical_cast.hpp>

#include "utils/debug_log.h"

#include "jpeg/cpu/cpu_jpeg.h"
#include "jpeg/cuda/cuda_jpeg.cuh"
#include "jpeg/cuda/ohmura/gpu_jpeg.cuh"

using namespace std;
using namespace util;

bool parse_arg(int argc, char *argv[]);
void cpu_main(const string &file_name, const string &out_file_name);
void gpu_main(const string &file_name, const string &out_file_name);
void cuda_main(const string &file_name, const string &out_file_name, size_t block_width, size_t block_height,
	int quarity, int test);

static string program_name;
static string infile_name;
static string outfile_name;
static size_t block_width = 0;
static size_t block_height = 0;
static int quarity = 80;
static int test = 0;
static int type = 0;

int main(int argc, char *argv[]) {
	if (!parse_arg(argc, argv))
		return 0;

	switch (type) {
	case 0:
		cuda_main(infile_name, outfile_name, block_width, block_height, quarity, test);
		break;

	case 1:
		gpu_main(infile_name, outfile_name);
		break;

	case 2:
		cpu_main(infile_name, outfile_name);
		break;

	default:
		cuda_main(infile_name, outfile_name, block_width, block_height, quarity, test);
		break;
	}

	return 0;
}

// コマンドライン引数解析
// command <input_filename>
// [output_filename] [-b <block_width> <block_height>]
// [-log <true|false>] [-logfile <true|false>]
// [-q <quarity[1,100]>]
//
// command <-h | --help>
bool parse_arg(int argc, char *argv[]) {
	program_name = argv[0];

	if (argc == 1) {
		cout << "command "
			"<input_filename [output_filename]> [-b <block_width> <block_height>] "
			"[-log <true|false>] [-logfile <true|false>] [-q <quarity[1,100]>]"
			"\n"
			"command <-h | --help>" << endl;
		return false;
	}

	infile_name = argv[1];

	if (infile_name == "-h" || infile_name == "--help") {
		cout << "command "
			"<input_filename [output_filename]> [-b <block_width> <block_height>] "
			"[-log <true|false>] [-logfile <true|false>] [-q <quarity[1,100]>]"
			"\n"
			"command <-h | --help>" << endl;
		return false;
	}

	for (int i = 2; i < argc; ++i) {
		string arg(argv[i]);
		if (arg == "-b") {
			if (argc - (i + 1) < 2) {
				cout << "Please input block size." << endl;
				return false;
			}
			block_width = boost::lexical_cast<size_t>(argv[++i]);
			block_height = boost::lexical_cast<size_t>(argv[++i]);
		} else if (arg == "-log") {
			if (argc - ++i < 1) {
				cout << "Please input flag." << endl;
				return false;
			}
			string subarg(argv[i]);
			bool flag = false;
			if (subarg == "true")
				flag = true;
			DebugLog::enablePrint(flag);
		} else if (arg == "-logfile") {
			if (argc - ++i < 1) {
				cout << "Please input flag." << endl;
				return false;
			}
			string subarg(argv[i]);
			bool flag = false;
			if (subarg == "true")
				flag = true;
			DebugLog::enableExport(flag);
		} else if (arg == "-q") {
			if (argc - ++i < 1) {
				cout << "Please input quarity." << endl;
				return false;
			}
			quarity = boost::lexical_cast<int>(argv[i]);
		} else if (arg == "-test") {
			if (argc - ++i < 1) {
				cout << "Please input test num." << endl;
				return false;
			}
			test = boost::lexical_cast<int>(argv[i]);
		} else if (arg == "-type") {
			if (argc - ++i < 1) {
				cout << "Please input type num." << endl;
				return false;
			}
			type = boost::lexical_cast<int>(argv[i]);
		} else {
			if (i == 2 && arg[0] != '-') {
				if (arg.substr(arg.length() - 4, arg.length()) != ".bmp") {
					outfile_name = arg + ".bmp";
				} else {
					outfile_name = arg;
				}
			} else {
				cout << "The option is N/A or undefined." << endl;
				return false;
			}
		}
	}
	return true;
}

