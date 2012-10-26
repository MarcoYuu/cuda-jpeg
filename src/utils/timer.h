﻿#pragma once

#include <vector>

//-----------------------------------------------------------------------------------------------
//時間計測用のインタフェース提供
//-----------------------------------------------------------------------------------------------
class ICountTime {
public:
	virtual ~ICountTime() {
	}
	virtual double getTimeInSeconds() const = 0;
};

//-----------------------------------------------------------------------------------------------
// StopWatch :stopではラップは刻まれないlap->stopで。
//-----------------------------------------------------------------------------------------------
class StopWatch {
public:
	typedef std::vector<double> LapList;
	enum Mode {
		CPU_OPTIMUM, C_STD, OTHER
	};

	explicit StopWatch(Mode mode);
	~StopWatch();

	void start();
	void stop();
	void lap();
	void clear();

	size_t getLapCount() const;
	double getTotalTime() const;
	double getLastElapsedTime() const;
	const LapList& getLapList() const;

private:
	ICountTime *m_counter;
	double m_prev_time;
	double m_elapse_time;
	LapList m_lap;
	Mode m_mode;

	StopWatch(StopWatch &rhs);
	void operator=(StopWatch);
};
