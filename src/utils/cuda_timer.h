/*
 * cuda_timer.h
 *
 *  Created on: 2012/09/26
 *      Author: momma
 */

#ifndef CUDA_TIMER_H_
#define CUDA_TIMER_H_

#include <vector>
#include <cuda_runtime.h>

//-----------------------------------------------------------------------------------------------
// StopWatch :stopではラップは刻まれないlap->stopで。
//-----------------------------------------------------------------------------------------------
class CudaStopWatch {
public:
	typedef std::vector<double> LapList;
	enum Mode {
		CPU_OPTIMUM, C_STD, OTHER
	};

	explicit CudaStopWatch();
	~CudaStopWatch();

	void start();
	void stop();
	void lap();
	void clear();

	int getLapCount() const;
	double getTotalTime() const;
	double getLastElapsedTime() const;
	const LapList& getLapList() const;

private:
	cudaEvent_t m_start, m_end;
	float m_elapse_time;
	LapList m_lap;
	Mode m_mode;

	CudaStopWatch(CudaStopWatch &rhs);
	void operator=(CudaStopWatch);
};

#endif /* CUDA_TIMER_H_ */
