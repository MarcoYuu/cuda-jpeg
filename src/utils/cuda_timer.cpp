/*
 * cuda_timer.cpp
 *
 *  Created on: 2012/09/26
 *      Author: momma
 */

#include "cuda_timer.h"

namespace util {
	namespace cuda {
		CudaStopWatch::CudaStopWatch() {
			cudaEventCreate(&m_start);
			cudaEventCreate(&m_end);
		}

		CudaStopWatch::~CudaStopWatch() {
			cudaEventDestroy(m_start);
			cudaEventDestroy(m_end);
		}

		void CudaStopWatch::start() {
			cudaEventRecord(m_start, 0);
		}

		void CudaStopWatch::stop() {
			cudaEventRecord(m_end, 0);
			cudaEventSynchronize(m_end);
			cudaEventElapsedTime(&m_elapse_time, m_start, m_end);
			m_elapse_time /= 1000.0;
		}

		void CudaStopWatch::lap() {
			cudaEventRecord(m_end, 0);
			cudaEventSynchronize(m_end);
			cudaEventElapsedTime(&m_elapse_time, m_start, m_end);
			m_elapse_time /= 1000.0;
			m_lap.push_back(m_elapse_time);

			cudaEventRecord(m_start, 0);
		}

		void CudaStopWatch::clear() {
			m_elapse_time = 0;
			m_lap.clear();
		}

		int CudaStopWatch::getLapCount() const {
			return m_lap.size();
		}

		double CudaStopWatch::getTotalTime() const {
			double r = 0;
			for (int i = 0; i < m_lap.size(); ++i)
				r += m_lap[i];
			return r;
		}

		double CudaStopWatch::getLastElapsedTime() const {
			return m_elapse_time;
		}

		const CudaStopWatch::LapList& CudaStopWatch::getLapList() const {
			return m_lap;
		}
	} // namespace cuda
} // namespace utils

