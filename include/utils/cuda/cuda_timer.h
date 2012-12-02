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

#include <utils/timer.h>

namespace util {
	namespace cuda {
		//-----------------------------------------------------------------------------------------------
		// StopWatch :stopではラップは刻まれないlap->stopで。
		//-----------------------------------------------------------------------------------------------
		/**
		 * @brief CUDAの同期関数を用いたGPU時間計測クラス
		 *
		 * - stopではラップは刻まれない。リセットせず特定区間の経過時間をgetElapseするには、lap->stopで。
		 *
		 * @author yuumomma
		 * @version 1.0
		 */
		class CudaStopWatch: public WatchInterface {
		public:
			/**
			 * @brief コンストラクタ
			 */
			explicit CudaStopWatch();
			/**
			 * @brief デストラクタ
			 */
			~CudaStopWatch();

			/**
			 * @brief 計測開始
			 */
			void start();
			/**
			 * @brief 一時停止
			 */
			void stop();
			/**
			 * @brief ラップタイムを記録する
			 */
			void lap();
			/**
			 * @brief リセット
			 */
			void clear();

			/**
			 * @brief 記録されているラップカウント数を返す
			 *
			 * @return ラップ数
			 */
			size_t getLapCount() const;
			/**
			 * @brief 現在までの総経過時間を取得する
			 *
			 * @return
			 */
			double getTotalTime() const;
			/**
			 * @brief 直前の差分時間を返す
			 *
			 * - stop()で止めた場合、直前のstart()またはlap()からの時間である
			 * - lap()を使った場合、直前のstart()またはlap()からの時間である
			 * - lap()しない場合返り値はgetTotalTime()と同じである
			 *
			 * @return 差分時間
			 */
			double getLastElapsedTime() const;
			/**
			 * @brief ラップタイムリストを返す
			 *
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
