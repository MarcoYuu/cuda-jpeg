/*
 * main.cpp
 *
 *  Created on: 2012/09/21
 *      Author: Yuu Momma
 */

#include <cstdlib>
#include <string>
#include <iostream>

void parse_arg(int argc, char *argv[], std::string &in_file, std::string &out_file);
void cpu_exec(const std::string &file_name, const std::string &out_file_name);
void gpu_exec(const std::string &file_name, const std::string &out_file_name);

int main(int argc, char *argv[]) {
	std::string file_name, out_file_name;
	parse_arg(argc, argv, file_name, out_file_name);

	cpu_exec(file_name, out_file_name);
	gpu_exec(file_name, out_file_name);

	return 0;
}

// コマンドライン引数解析
void parse_arg(int argc, char *argv[], std::string &in_file, std::string &out_file) {
	if (argc == 3) {
		in_file = argv[1];
		out_file = argv[2];
	} else {
		std::cout << "Please input source file." << std::endl;
		abort();
	}
}

