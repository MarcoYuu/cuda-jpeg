/*
 * type_definitions.h
 *
 *  Created on: 2012/09/22
 *      Author: Yuu Momma
 */

#ifndef CUDA_JPEG_TYPES_H_
#define CUDA_JPEG_TYPES_H_

#include <vector>

namespace util {
	typedef unsigned char byte;
	typedef unsigned int u_int;

	typedef std::vector<byte> ByteBuffer;
	typedef std::vector<int> IntBuffer;
}  // namespace util

#endif /* CUDA_JPEG_TYPES_H_ */
