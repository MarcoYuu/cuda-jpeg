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
#include "jpeg/ohmura/gpu_jpeg.cuh"

using namespace std;
using namespace util;

void parse_arg(int argc, char *argv[]);
void cpu_main(const string &file_name, const string &out_file_name);
void gpu_main(const string &file_name, const string &out_file_name);
void cuda_main(const string &file_name, const string &out_file_name, size_t block_width, size_t block_height, int quarity);

string program_name;
string infile_name;
string outfile_name;
size_t block_width = 32;
size_t block_height = 32;
int quarity = 80;

int main(int argc, char *argv[]) {
	parse_arg(argc, argv);

	//cpu_main(infile_name, outfile_name);
	//gpu_main(infile_name, outfile_name);
	cuda_main(infile_name, outfile_name, block_width, block_height, quarity);

	return 0;
}

// コマンドライン引数解析
// command <input_filename>
// [output_filename] [-b <block_width> <block_height>]
// [-log <true|false>] [-logfile <true|false>]
// [-q <quarity[1,100]>]
// command <-h | --help>
void parse_arg(int argc, char *argv[]) {
	program_name = argv[0];
	if (argc == 1) {
		cout << "Please input source file." << endl;
		exit(0);
	}

	infile_name = argv[1];
	outfile_name = argv[1];

	for (int i = 2; i < argc; ++i) {
		string arg(argv[i]);
		if (arg == "-h" || arg == "--help") {
			cout << "command "
				"<input_filename> "
				"[output_filename] "
				"[-b <block_width> <block_height>] "
				"[-log <true|false>] "
				"[-logfile <true|false>]"
				"\n"
				"command <-h | --help>" << endl;
			exit(0);
		} else if (arg == "-b") {
			if (argc - (i + 1) < 2) {
				cout << "Please input block size." << endl;
				exit(0);
			}
			block_width = boost::lexical_cast<size_t>(argv[++i]);
			block_height = boost::lexical_cast<size_t>(argv[++i]);
		} else if (arg == "-log") {
			string subarg(argv[++i]);
			bool flag = false;
			if (subarg == "true")
				flag = true;
			DebugLog::enablePrint(flag);
		} else if (arg == "-logfile") {
			string subarg(argv[++i]);
			bool flag = false;
			if (subarg == "true")
				flag = true;
			DebugLog::enableExport(flag);
		} else if (arg == "-q") {
			quarity = boost::lexical_cast<int>(argv[++i]);
		} else {
			outfile_name = argv[i];
		}
	}
}

