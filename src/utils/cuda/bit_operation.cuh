/*
 * bit_operation.cuh
 *
 *  Created on: 2012/12/01
 *      Author: yuumomma
 */

#ifndef BIT_OPERATION_HPP_
#define BIT_OPERATION_HPP_

#include "../type_definitions.h"

namespace util {
	/**
	 * 立っているbit数を数える
	 *
	 * @param value
	 * @return
	 */
	__host__ __device__ inline int CountBits(byte value) {
		unsigned count = (value & 0x55) + ((value >> 1) & 0x55);
		count = (count & 0x33) + ((count >> 2) & 0x33);
		return (count & 0x0f) + ((count >> 4) & 0x0f);
	}

	__host__ __device__ inline int CountBits(byte2 value) {
		unsigned short count = (value & 0x5555) + ((value >> 1) & 0x5555);
		count = (count & 0x3333) + ((count >> 2) & 0x3333);
		count = (count & 0x0f0f) + ((count >> 4) & 0x0f0f);
		return (count & 0x00ff) + ((count >> 8) & 0x00ff);
	}

	__host__ __device__ inline int CountBits(byte4 value) {
		unsigned count = (value & 0x55555555) + ((value >> 1) & 0x55555555);
		count = (count & 0x33333333) + ((count >> 2) & 0x33333333);
		count = (count & 0x0f0f0f0f) + ((count >> 4) & 0x0f0f0f0f);
		count = (count & 0x00ff00ff) + ((count >> 8) & 0x00ff00ff);
		return (count & 0x0000ffff) + ((count >> 16) & 0x0000ffff);
	}

	__host__ __device__ inline int EffectiveBits(byte value) {
		value |= (value >> 1);
		value |= (value >> 2);
		value |= (value >> 4);
		return CountBits(value);
	}

	__host__ __device__ inline int EffectiveBits(byte2 value) {
		value |= (value >> 1);
		value |= (value >> 2);
		value |= (value >> 4);
		value |= (value >> 8);
		return CountBits(value);
	}

	__host__ __device__ inline int EffectiveBits(byte4 value) {
		value |= (value >> 1);
		value |= (value >> 2);
		value |= (value >> 4);
		value |= (value >> 8);
		value |= (value >> 16);
		return CountBits(value);
	}
}  // namespace util

#endif /* BIT_OPERATION_HPP_ */
