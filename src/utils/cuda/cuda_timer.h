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

namespace util {
	namespace cuda {
		//-----------------------------------------------------------------------------------------------
		// StopWatch :stopではラップは刻まれないlap->stopで。
		//-----------------------------------------------------------------------------------------------
		/**
		 * CUDAの同期関数を用いたGPU時間計測クラス
		 *
		 * -stopではラップは刻まれない。リセットせず特定区間の経過時間をgetElapseするには、lap->stopで。
		 *
		 * @author yuumomma
		 * @version 1.0
		 */
		class CudaStopWatch {
		public:
			typedef std::vector<double> LapList;

			/**
			 * コンストラクタ
			 */
			explicit CudaStopWatch();
			/**
			 * デストラクタ
			 */
			~CudaStopWatch();

			/**
			 * 計測開始
			 */
			void start();
			/**
			 * 一時停止
			 */
			void stop();
			/**
			 * ラップタイムを記録する
			 */
			void lap();
			/**
			 * リセット
			 */
			void clear();

			/**
			 * 記録されているラップカウント数を返す
			 * @return ラップ数
			 */
			int getLapCount() const;
			/**
			 * 現在までの総経過時間を取得する
			 * @return
			 */
			double getTotalTime() const;
			/**
			 * 直前の差分時間を返す
			 * -stop()で止めた場合、直前のstart()またはlap()からの時間である
			 * -lap()を使った場合、直前のstart()またはlap()からの時間である
			 * -したがって、lap()しない場合返り値はgetTotalTime()と同じである
			 * @return 差分時間
			 */
			double getLastElapsedTime() const;
			/**
			 * ラップタイムリストを返す
			 * @return リスト
			 */
			const LapList& getLapList() const;

		private:
			cudaEvent_t m_start, m_end;
			float m_elapse_time;
			LapList m_lap;

			CudaStopWatch(CudaStopWatch &rhs);
			void operator=(CudaStopWatch);
		};
	} // namespace cuda
} // namespace utils

#endif /* CUDA_TIMER_H_ */
