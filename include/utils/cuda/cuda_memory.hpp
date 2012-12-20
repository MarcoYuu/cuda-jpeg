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
#include <cstring>
#include <cassert>

#include <utils/debug_log.h>

namespace util {
	namespace cuda {
		/**
		 * @brief CUDAメモリ簡易管理
		 *
		 * デバイスに書き込むだけでホストから読み出す必要はないとき
		 *
		 * @author momma
		 */
		template<class T>
		class device_memory {
		private:
			T* device_mem_;
			size_t size_;

			device_memory(const device_memory<T>&);
			device_memory<T>& operator=(const device_memory<T>&);

		public:
			typedef T* iterator;
			typedef const T* const_iterator;

			virtual iterator begin() {
				return device_mem_;
			}

			virtual const_iterator begin() const {
				return device_mem_;
			}

			virtual iterator end() {
				return device_mem_ + size_;
			}

			virtual const_iterator end() const {
				return device_mem_ + size_;
			}

		public:
			/**
			 * @brief コンストラクタ
			 *
			 * @param size [in] Tの個数
			 */
			device_memory(size_t size) :
				size_(size) {
				// デバイスメモリの確保
				cudaMalloc((void**) &device_mem_, sizeof(T) * size);
			}

			/**
			 * @brief コンストラクタ
			 *
			 * @param size [in] Tの個数
			 * @param data [in] 初期化データ
			 */
			device_memory(const T* data, size_t size) :
				size_(size) {

				// デバイスメモリの確保とコピー
				cudaMalloc((void**) &device_mem_, sizeof(T) * size);
				cudaMemcpy(device_mem_, data, sizeof(T) * size, cudaMemcpyHostToDevice);
			}

			/**
			 * @brief デストラクタ
			 */
			virtual ~device_memory() {
				if (device_mem_ != NULL)
					cudaFree(device_mem_);

#ifdef DEBUG
				DebugLog::log("device_memory::~device_memory()");
#endif
			}

			/**
			 * @brief メモリゼロクリア
			 */
			virtual void fill_zero() {
				cudaMemset(device_mem_, 0, sizeof(T) * size_);
			}

			/**
			 * @brief リサイズ
			 *
			 * 二引数目をtrueにすると再アロケーション発生確定.
			 * 拡張時及び二引数目true時に中身は破棄される.
			 *
			 * @param size 変更サイズ
			 * @param force より小さくする際に、現在のバッファを完全に破棄するかどうか
			 */
			virtual void resize(size_t size, bool force = false) {
				if (force || size_ < size) {
					if (device_mem_ != NULL) {
						cudaFree(device_mem_);
					}
					cudaMalloc((void**) &device_mem_, sizeof(T) * size);
				}
				size_ = size;
			}

			/**
			 * @brief サイズを返す
			 *
			 * Tの個数を取得する。バイト数はsizeof(T)*size()。
			 *
			 * @return Tの個数
			 */
			size_t size() const {
				return size_;
			}

			/**
			 * @brief デバイスメモリに書き込む
			 *
			 * @param data [in] 書き込み元
			 * @param size [in] 書き込みサイズ
			 * @param offset [in] 書き込み先の先頭までのオフセット
			 */
			void write_device(const T* data, size_t size, size_t offset = 0) {
				assert(size + offset <= size_);
				cudaMemcpy(device_mem_ + sizeof(T) * offset, data, sizeof(T) * size, cudaMemcpyHostToDevice);
			}

			/**
			 * @brief 生データへのポインタ取得
			 */
			T* device_data() {
				return device_mem_;
			}
			/**
			 * @brief 生データへのポインタ取得
			 */
			const T* device_data() const {
				return device_mem_;
			}

			/**
			 * @brief ホストメモリにコピーする
			 *
			 * デバイスメモリのデータをホストメモリに転送する
			 *
			 * @param host_mem [out] 書き込み先
			 * @param size [in] 書き込みサイズ
			 */
			void copy_to_host(T* host_mem, size_t size, size_t offset = 0) {
				assert(size <= size_);
				cudaMemcpy(host_mem, device_mem_ + offset, sizeof(T) * size, cudaMemcpyDeviceToHost);
			}
		};

		/**
		 * @brief CUDAメモリ簡易管理
		 *
		 * @author momma
		 */
		template<class T>
		class cuda_memory: public device_memory<T> {
		private:
			typedef device_memory<T> base;
			T* host_mem_;

			cuda_memory(const cuda_memory<T>&);
			cuda_memory<T>& operator=(const cuda_memory<T>&);

		public:
			typename base::iterator begin() {
				return host_mem_;
			}

			typename base::const_iterator begin() const {
				return host_mem_;
			}

			typename base::iterator end() {
				return host_mem_ + base::size();
			}

			typename base::const_iterator end() const {
				return host_mem_ + base::size();
			}

		public:
			/**
			 * @brief コンストラクタ
			 *
			 * @param size [in] Tの個数
			 */
			cuda_memory(size_t size) :
				device_memory<T>(size) {
				// ホストメモリの確保
				host_mem_ = new T[size];
			}

			/**
			 * @brief コンストラクタ
			 *
			 * @param size [in] Tの個数
			 * @param data [in] 初期化データ
			 * @param copy_to_device [in] デバイスにコピーするかどうか
			 */
			cuda_memory(const T* data, size_t size, bool copy_to_device) :
				device_memory<T>(data, size) {
				// ホストメモリの確保と初期化
				host_mem_ = new T[size];
				memcpy(host_mem_, data, sizeof(T) * size);

				// デバイスメモリのコピー
				if (copy_to_device) {
					//write_device(data, size);
					sync_to_device();
				}
			}

			/**
			 * @brief デストラクタ
			 */
			virtual ~cuda_memory() {
				delete[] host_mem_;

#ifdef DEBUG
				DebugLog::log("cuda_memory::~cuda_memory()");
#endif
			}

			/**
			 * @brief メモリゼロクリア
			 */
			void fill_zero() {
				memset(host_mem_, 0, sizeof(T) * this->size());
				base::fill_zero();
			}

			/**
			 * @brief メモリゼロクリア
			 */
			void fill_zero_host() {
				memset(host_mem_, 0, sizeof(T) * this->size());
			}

			/**
			 * @brief メモリゼロクリア
			 */
			void fill_zero_device() {
				base::fill_zero();
			}

			/**
			 * @brief リサイズする
			 *
			 * 二引数目をtrueにすると再アロケーション発生確定.
			 * 拡張時及び二引数目true時に中身は破棄される.
			 *
			 * @param size 変更サイズ
			 * @param force より小さくする際に、現在のバッファを完全に破棄するかどうか
			 */
			void resize(size_t size, bool force = false) {
				if (force || base::size() < size) {
					delete[] host_mem_;
					host_mem_ = new T[size];
				}
				base::resize(size, force);
			}

			/**
			 * @brief ホストメモリに書き込む
			 *
			 * @param data [in] 書き込み元
			 * @param size [in] 書き込みサイズ
			 * @param offset [in] 書き込み先の先頭までのオフセット
			 */
			void write_host(const T* data, size_t size, size_t offset = 0) {
				assert(size <= base::size());
				memcpy(host_mem_ + offset, data, sizeof(T) * size);
			}

			/**
			 * @brief 生データへのポインタ取得
			 */
			T* host_data() {
				return host_mem_;
			}
			/**
			 * @brief 生データへのポインタ取得
			 */
			const T* host_data() const {
				return host_mem_;
			}

			/**
			 * @brief デバイスメモリの中身を同期させる
			 *
			 * ホストメモリのデータをデバイスメモリに転送する
			 */
			void sync_to_device() {
				cudaMemcpy(base::device_data(), host_mem_, sizeof(T) * base::size(), cudaMemcpyHostToDevice);
			}

			/**
			 * @brief ホストメモリの中身を同期させる
			 *
			 * デバイスメモリのデータをホストメモリに転送する
			 */
			void sync_to_host() {
				cudaMemcpy(host_mem_, base::device_data(), sizeof(T) * base::size(), cudaMemcpyDeviceToHost);
			}

			/**
			 * @brief インデクサ
			 *
			 * @param index
			 * @return ホストメモリ
			 */
			T& operator[](size_t index) {
				assert(index < base::size());
				return host_mem_[index];
			}
			/**
			 * @brief インデクサ
			 *
			 * @param index
			 * @return ホストメモリ
			 */
			const T& operator[](size_t index) const {
				assert(index < base::size());
				return host_mem_[index];
			}
		};
	} // namespace cuda
} // namespace util

#endif /* CUDAMEMORY_HPP_ */
