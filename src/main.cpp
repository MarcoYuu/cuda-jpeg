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

#include "jpeg/cuda/cuda_jpeg.cuh"

void parse_arg(int argc, char *argv[]);
//void cpu_exec(const std::string &file_name, const std::string &out_file_name);
//void gpu_exec(const std::string &file_name, const std::string &out_file_name);
void cuda_exec(const std::string &file_name, const std::string &out_file_name, size_t block_width,
	size_t block_height);

std::string program_name;
std::string infile_name;
std::string outfile_name;
size_t block_width = 32;
size_t block_height = 32;

int main(int argc, char *argv[]) {
	parse_arg(argc, argv);

	//cpu_exec(file_name, out_file_name);
	//gpu_exec(file_name, out_file_name);
	cuda_exec(infile_name, outfile_name, block_width, block_height);

	return 0;
}

// コマンドライン引数解析
// command <input_filename> [output_filename] [-b <block_width> <block_height>]
// command <-h | --help>
void parse_arg(int argc, char *argv[]) {
	program_name = argv[0];
	std::cout << "program_name: " << program_name << std::endl;

	if (argc == 1) {
		std::cout << "Please input source file." << std::endl;
		exit(0);
	}

	infile_name = argv[1];
	outfile_name = argv[1];

	for (int i = 2; i < argc; ++i) {
		std::string arg(argv[i]);
		if (arg == "-h" || arg == "--help") {
			std::cout
				<< "command <input_filename> [output_filename] [-b <block_width> <block_height>]\n"
					"command <-h | --help>" << std::endl;
			exit(0);
		} else if (arg == "-b") {
			if (argc - (i + 1) < 2) {
				std::cout << "Please input block size." << std::endl;
				exit(0);
			}
			block_width = boost::lexical_cast<size_t>(argv[++i]);
			block_height = boost::lexical_cast<size_t>(argv[++i]);
		} else {
			outfile_name = argv[i];
		}
	}
	std::cout << "infile_name: " << infile_name << std::endl;
	std::cout << "outfile_name: " << outfile_name << std::endl;
	std::cout << "block_width: " << block_width << std::endl;
	std::cout << "block_height: " << block_height << std::endl;
}

