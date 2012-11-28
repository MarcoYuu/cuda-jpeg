#pragma once

#include <vector>

namespace util {
	//-----------------------------------------------------------------------------------------------
	//時間計測用のインタフェース提供
	//-----------------------------------------------------------------------------------------------
	/**
	 * 時間計測用のインタフェース
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

	class Watch {
	public:
		typedef std::vector<double> LapList;
		virtual ~Watch() {
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
	// StopWatch :stopではラップは刻まれないlap->stopで。
	//-----------------------------------------------------------------------------------------------
	/**
	 * それなりに最適な時間測定関数を利用した時間計測クラス
	 *
	 * @author yuumomma
	 * @version 1.0
	 */
	class StopWatch: public Watch {
	public:
		/**
		 * 時間計測に用いる手法の選択フラグ
		 * -CPU_OPTIMUM:WindowsとLinuxでそれなりに精度の高いもの
		 * -C_STD:clock()
		 * -OTHER:未定義
		 *
		 * @author yuumomma
		 * @version 1.0
		 */
		enum Mode {
			CPU_OPTIMUM, C_STD, OTHER
		};

		/**
		 * コンストラクタ
		 * @param mode モード
		 * @sa StopWatch::Mode
		 */
		explicit StopWatch(Mode mode = CPU_OPTIMUM);
		/**
		 * デストラクタ
		 */
		~StopWatch();

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
		size_t getLapCount() const;
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
		struct impl;
		impl *m_impl;

		StopWatch(StopWatch &rhs);
		void operator=(StopWatch &);
	};
} // namespace util
