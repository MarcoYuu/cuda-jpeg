/*
 * utility.hpp
 *
 *  Created on: 2012/12/9
 *      Author: momma
 */

#ifndef UTILITY_H_
#define UTILITY_H_

#include <algorithm>

namespace util {
	template<class T>
	T gcd(T x, T y) {
		if (x < y) {
			std::swap(x, y);
		}
		int result = x % y;
		// ユーグリッドの互除法
		while (result != 0) {
			x = y;
			y = result;
			result = x % y;
		}
		return y;
	}
} // namespace util

#endif /* UTILITY_H_ */
