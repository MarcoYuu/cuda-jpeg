#include <ctime>
#include <numeric>

#if defined(_WIN32) | defined(_WIN64)
#pragma comment(lib, "winmm.lib")
#include <windows.h>
#include <mmsystem.h>
#elif defined(__linux__)
#include <sys/time.h>
#include <sys/resource.h>
#endif

#include "timer.h"

class StdTimeCounter;
#if defined(_WIN32) | defined(_WIN64)
class MultiMediaCounterp;
class ClockFreqCounter;
#elif defined(__linux__)
class UnixTimeCounter;
class ResourceTimeCounter;
#endif

//-----------------------------------------------------------------------------------------------
//clock()関数を用いた時間計測クラス
//-----------------------------------------------------------------------------------------------
class StdTimeCounter: public ICountTime {
public:
	double getTimeInSeconds() const {
		return (double) clock() / (double) CLOCKS_PER_SEC;
	}
};

#if defined(_WIN32) | defined(_WIN64)

//-----------------------------------------------------------------------------------------------
//マルチメディタイマを用いた時間計測クラス
//-----------------------------------------------------------------------------------------------
class MultiMediaCounter : public ICountTime {
public:
	MultiMediaCounter() {
		timeBeginPeriod(1);
	}
	~MultiMediaCounter() {
		timeEndPeriod(1);
	}
	double getTimeInSeconds() const {
		return timeGetTime() / 1000.0;
	}
};

//-----------------------------------------------------------------------------------------------
//FrequencyTimerを用いた時間計測クラス
//-----------------------------------------------------------------------------------------------
class ClockFreqCounter : public ICountTime {
private:
	double mCPUFreqency;

public:
	ClockFreqCounter()
	: mCPUFreqency(0.0) {
		LARGE_INTEGER count;
		LARGE_INTEGER freq;
		QueryPerformanceCounter(&count);
		QueryPerformanceFrequency(&freq);
		mCPUFreqency = (double)freq.QuadPart;
	}

	double getTimeInSeconds() const {
		LARGE_INTEGER count;
		QueryPerformanceCounter(&count);
		return count.QuadPart / mCPUFreqency;
	}

	static bool isAvailable() {
		LARGE_INTEGER tmp;
		return ((QueryPerformanceCounter(&tmp) == 0) ? false : true);
	}
};

#elif defined(__linux__)

//-----------------------------------------------------------------------------------------------
//gettimeofday()関数を用いた時間計測クラス
//-----------------------------------------------------------------------------------------------
class UnixTimeCounter: public ICountTime {
public:
	double getTimeInSeconds() const {
		timeval s;
		gettimeofday(&s, NULL);
		return s.tv_sec + s.tv_usec * 0.000001;
	}
};

//-----------------------------------------------------------------------------------------------
//getrusage()関数を用いた時間計測クラス
//-----------------------------------------------------------------------------------------------
class ResourceTimeCounter: public ICountTime {
public:
	double getTimeInSeconds() const {
		rusage t;
		getrusage(RUSAGE_SELF, &t);
		return t.ru_utime.tv_sec + (double) t.ru_utime.tv_usec * 0.000001;
	}
};

#endif

//-----------------------------------------------------------------------------------------------
// StopWatch
//-----------------------------------------------------------------------------------------------
ICountTime* CreateCounter(StopWatch::Mode mode) {
	switch (mode) {
	case StopWatch::CPU_OPTIMUM:
#if defined(_WIN32) | defined(_WIN64)
		if(ClockFreqCounter::isAvailable())
		return new ClockFreqCounter();
		else
		return new MultiMediaCounter();
#elif defined(__linux__)
		return new UnixTimeCounter();
		//return new ResourceTimeCounter();
#endif
	case StopWatch::C_STD:
		return new StdTimeCounter();
	}
	return NULL;
}

StopWatch::StopWatch(Mode mode) :
	m_counter(CreateCounter(mode)),
	m_prev_time(0.0),
	m_elapse_time(0.0),
	m_lap(),
	m_mode(mode) {
}
StopWatch::~StopWatch() {
	delete m_counter;
}

void StopWatch::start() {
	m_prev_time = m_counter->getTimeInSeconds();
}
void StopWatch::stop() {
	m_elapse_time += m_counter->getTimeInSeconds() - m_prev_time;
}
void StopWatch::lap() {
	double t = m_counter->getTimeInSeconds();
	m_elapse_time = t - m_prev_time;
	m_prev_time = t;
	m_lap.push_back(m_elapse_time);
}
void StopWatch::clear() {
	m_prev_time = 0;
	m_elapse_time = 0;
	m_lap.clear();
}

double StopWatch::getLastElapsedTime() const {
	return m_elapse_time;
}

double StopWatch::getTotalTime() const {
	double r = 0;
	for (int i = 0; i < m_lap.size(); ++i)
		r += m_lap[i];
	return r;
}

size_t StopWatch::getLapCount() const {
	return m_lap.size();
}

const StopWatch::LapList& StopWatch::getLapList() const {
	return m_lap;
}
