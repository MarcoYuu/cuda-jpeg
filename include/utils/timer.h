#pragma once

#include <vector>

namespace util {
	//-----------------------------------------------------------------------------------------------
	//時間計測用のインタフェース提供
	//-----------------------------------------------------------------------------------------------
	/**
	 * @brief 時間計測用のインタフェース
	 *
	 * @author yuumomma
	 * @version 1.0
	 */
	class ICountTime {
	public:
		virtual ~ICountTime() {
		}
		virtual double getTimeInSeconds() const = 0;
	};

	/**
	 * @brief タイマーインタフェース
	 *
	 * @author yuumomma
	 * @version 1.0
	 */
	class WatchInterface {
	public:
		typedef std::vector<double> LapList;
		virtual ~WatchInterface() {
		}
		virtual void start() =0;
		virtual void stop() =0;
		virtual void lap() =0;
		virtual void clear() =0;

		virtual size_t getLapCount() const =0;
		virtual double getTotalTime() const =0;
		virtual double getLastElapsedTime() const =0;
		virtual const LapList& getLapList() const =0;
	};
	//-----------------------------------------------------------------------------------------------
	// StopWatch :
	//-----------------------------------------------------------------------------------------------
	/**
	 * @brief ストップウォッチクラス
	 *
	 * それなりに最適な時間測定関数を利用した時間計測クラス
	 * stopではラップは刻まれないlap->stopで。
	 *
	 * @author yuumomma
	 * @version 1.0
	 */
	class StopWatch: public WatchInterface {
	public:
		/**
		 * @brief 時間計測に用いる手法の選択フラグ
		 *
		 * - CPU_OPTIMUM:WindowsとLinuxでそれなりに精度の高いもの
		 * - C_STD:clock()
		 * - OTHER:未定義
		 *
		 * @author yuumomma
		 * @version 1.0
		 */
		enum Mode {
			CPU_OPTIMUM, C_STD, OTHER
		};

		/**
		 * @brief コンストラクタ
		 *
		 * @param mode モード
		 * @sa StopWatch::Mode
		 */
		explicit StopWatch(Mode mode = CPU_OPTIMUM);
		/**
		 * @brief デストラクタ
		 */
		~StopWatch();

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
		 * - したがって、lap()しない場合返り値はgetTotalTime()と同じである
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
		struct impl;
		impl *m_impl;

		StopWatch(StopWatch &rhs);
		void operator=(StopWatch &);
	};
} // namespace util
