/*
 * utils.hpp
 *
 *  Created on: 2012/11/12
 *      Author: momma
 */

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <fstream>
#include <string>

namespace util {
	void dump_memory(void* data, size_t data_size, const std::string &filename) {
		std::ofstream ofs(filename.c_str(), std::ios::binary);
		ofs.write((char*) data, data_size);
	}
}

#endif /* UTILS_HPP_ */
