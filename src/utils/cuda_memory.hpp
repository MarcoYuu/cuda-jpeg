/*
 * cudamemory.h
 *
 *  Created on: 2012/10/15
 *      Author: momma
 */

#ifndef CUDAMEMORY_HPP_
#define CUDAMEMORY_HPP_

#include <cuda_runtime.h>
#include <cstdlib>

namespace util {
	namespace cuda {
		/**
		 * CUDAメモリ簡易管理
		 *
		 * デバイスに書き込むだけでホストから読み出す必要はないとき
		 *
		 * @author momma
		 */
		template<class T>
		class device_memory {
		private:
			T* _device_mem;
			size_t _size;

			device_memory(const device_memory<T>&);
			device_memory<T>& operator=(const device_memory<T>&);

		public:
			/**
			 * コンストラクタ
			 *
			 * @param size [in] Tの個数
			 */
			device_memory(size_t size) :
				_size(size) {
				// デバイスメモリの確保
				cudaMalloc((void**) &_device_mem, sizeof(T) * size);
			}

			/**
			 * コンストラクタ
			 *
			 * @param size [in] Tの個数
			 * @param data [in] 初期化データ
			 */
			device_memory(const T* data, size_t size) :
				_size(size) {

				// デバイスメモリの確保とコピー
				cudaMalloc((void**) &_device_mem, sizeof(T) * size);
				cudaMemcpy(_device_mem, data, sizeof(T) * size, cudaMemcpyHostToDevice);
			}

			/**
			 * デストラクタ
			 */
			virtual ~device_memory() {
				if (_device_mem != NULL)
					cudaFree(_device_mem);
			}

			/**
			 * メモリをゼロクリア
			 */
			virtual void fill_zero() {
				cudaMemset(_device_mem, 0, sizeof(T) * _size);
			}

			/**
			 * メモリサイズを返す
			 *
			 * Tの個数を取得する。バイト数はsizeof(T)*CudaMemory::size()。
			 *
			 * @return Tの個数
			 */
			size_t size() const {
				return _size;
			}

			/**
			 * デバイスメモリに書き込む
			 *
			 * @param data [in] 書き込み元
			 * @param size [in] 書き込みサイズ
			 */
			void write_device(const T* data, size_t size) {
				cudaMemcpy(_device_mem, data, sizeof(T) * size, cudaMemcpyHostToDevice);
			}

			/**
			 * 生データへのポインタ取得
			 */
			T* device_data() {
				return _device_mem;
			}
			/**
			 * 生データへのポインタ取得
			 */
			const T* device_data() const {
				return _device_mem;
			}

			/**
			 * ホストメモリにコピーする
			 *
			 * デバイスメモリのデータをホストメモリに転送する
			 *
			 * @param host_mem [out] 書き込み先
			 * @param size [in] 書き込みサイズ
			 */
			void copy_host(T* host_mem, size_t size) {
				cudaMemcpy(host_mem, _device_mem, sizeof(T) * size, cudaMemcpyDeviceToHost);
			}
		};

		/**
		 * CUDAメモリ簡易管理
		 *
		 * @author momma
		 */
		template<class T>
		class cuda_memory: public device_memory<T> {
		private:
			typedef device_memory<T> base;
			T* _host_mem;

			cuda_memory(const cuda_memory<T>&);
			cuda_memory<T>& operator=(const cuda_memory<T>&);

		public:
			/**
			 * コンストラクタ
			 *
			 * @param size [in] Tの個数
			 */
			cuda_memory(size_t size) :
				device_memory<T>(size) {
				// ホストメモリの確保
				_host_mem = new T[size];
			}

			/**
			 * コンストラクタ
			 *
			 * @param size [in] Tの個数
			 * @param data [in] 初期化データ
			 * @param copy_to_device [in] デバイスにコピーするかどうか
			 */
			cuda_memory(const T* data, size_t size, bool copy_to_device) :
				device_memory<T>(data, size) {
				// ホストメモリの確保と初期化
				_host_mem = new T[size];
				memcpy(_host_mem, data, sizeof(T) * size);

				// デバイスメモリのコピー
				if (copy_to_device) {
					write_device(data, size);
				}
			}

			/**
			 * デストラクタ
			 */
			virtual ~cuda_memory() {
				delete[] _host_mem;
			}
			/**
			 * メモリをゼロクリア
			 */
			void fillZero() {
				memset(_host_mem, 0, sizeof(T) * this->size());
				base::fill_zero();
			}

			/**
			 * ホストメモリに書き込む
			 *
			 * @param data [in] 書き込み元
			 * @param size [in] 書き込みサイズ
			 */
			void write_host(const T* data, size_t size) {
				memcpy(_host_mem, data, sizeof(T) * size);
			}

			/**
			 * 生データへのポインタ取得
			 */
			T* host_data() {
				return _host_mem;
			}
			/**
			 * 生データへのポインタ取得
			 */
			const T* host_data() const {
				return _host_mem;
			}

			/**
			 * デバイスメモリの中身を同期させる
			 *
			 * ホストメモリのデータをデバイスメモリに転送する
			 */
			void sync_to_device() {
				cudaMemcpy(base::device_data(), _host_mem, sizeof(T) * base::size(),
					cudaMemcpyHostToDevice);
			}

			/**
			 * ホストメモリの中身を同期させる
			 *
			 * デバイスメモリのデータをホストメモリに転送する
			 */
			void sync_to_host() {
				cudaMemcpy(_host_mem, base::device_data(), sizeof(T) * base::size(),
					cudaMemcpyDeviceToHost);
			}

			/**
			 * インデクサ
			 *
			 * @param index
			 * @return ホストメモリ
			 */
			T& operator[](size_t index) {
				return _host_mem[index];
			}
			/**
			 * インデクサ
			 *
			 * @param index
			 * @return ホストメモリ
			 */
			const T& operator[](size_t index) const {
				return _host_mem[index];
			}
		};
	} // namespace cuda
} // namespace util

#endif /* CUDAMEMORY_HPP_ */
