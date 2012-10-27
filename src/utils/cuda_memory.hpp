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

/**
 * @brief CUDAメモリ簡易管理
 *
 * @author momma
 */
template<class T>
class CudaMemory {
private:
	T* _host_mem;
	T* _device_mem;
	size_t _size;

	CudaMemory(const CudaMemory<T>&);
	CudaMemory<T>& operator=(const CudaMemory<T>&);

public:
	/**
	 * @brief コンストラクタ
	 *
	 * @param size [in] Tの個数
	 */
	CudaMemory(size_t size) :
		_size(size) {
		// ホストメモリの確保
		_host_mem = new T[size];
		// デバイスメモリの確保
		cudaMalloc((void**) &_device_mem, sizeof(T) * size);
	}

	/**
	 * @brief コンストラクタ
	 *
	 * @param size [in] Tの個数
	 * @param data [in] 初期化データ
	 * @param copy_to_device [in] デバイスにコピーするかどうか
	 */
	CudaMemory(const T* data, size_t size, bool copy_to_device) :
		_size(size) {
		// ホストメモリの確保と初期化
		_host_mem = new T[size];
		memcpy(_host_mem, data, sizeof(T) * size);

		// デバイスメモリの確保とコピー
		cudaMalloc((void**) &_device_mem, sizeof(T) * size);
		if (copy_to_device) {
			cudaMemcpy(_device_mem, _host_mem, sizeof(T) * _size, cudaMemcpyHostToDevice);
		}
	}

	virtual ~CudaMemory() {
		delete[] _host_mem;
		if (_device_mem != NULL)
			cudaFree(_device_mem);
	}

	void fillZero() {
		memset(_host_mem, 0, sizeof(T) * _size);
		cudaMemset(_device_mem, 0, sizeof(T) * _size);
	}

	/**
	 * @brief デバイスメモリの中身を同期させる
	 *
	 * ホストメモリのデータをデバイスメモリに転送する
	 */
	void syncDeviceMemory() {
		cudaMemcpy(_device_mem, _host_mem, sizeof(T) * _size, cudaMemcpyHostToDevice);
	}

	/**
	 * @brief ホストメモリの中身を同期させる
	 *
	 * デバイスメモリのデータをホストメモリに転送する
	 */
	void syncHostMemory() {
		cudaMemcpy(_host_mem, _device_mem, sizeof(T) * _size, cudaMemcpyDeviceToHost);
	}

	void write_host(const T* data, size_t size) {
		memcpy(_host_mem, data, sizeof(T) * size);
	}

	void write_device(const T* data, size_t size) {
		cudaMemcpy(_device_mem, data, sizeof(T) * size, cudaMemcpyHostToDevice);
	}

	/**
	 * @brief メモリサイズを返す
	 *
	 * Tの個数を取得する。バイト数はsizeof(T)*CudaMemory::size()。
	 *
	 * @return Tの個数
	 */
	size_t size() const {
		return _size;
	}

	T& operator[](size_t index) {
		return _host_mem[index];
	}
	const T& operator[](size_t index) const {
		return _host_mem[index];
	}

	/**
	 * @brief 生データへのポインタ取得
	 */
	T* host_data() {
		return _host_mem;
	}
	const T* host_data() const {
		return _host_mem;
	}

	/**
	 * @brief 生データへのポインタ取得
	 */
	T* device_data() {
		return _device_mem;
	}
	const T* device_data() const {
		return _device_mem;
	}
};

/**
 * @brief CUDAメモリ簡易管理
 * 
 * デバイスに書き込むだけでホストから読み出す必要はないとき
 *
 * @author momma
 */
template<class T>
class DeviceMemory {
private:
	T* _device_mem;
	size_t _size;

	DeviceMemory(const CudaMemory<T>&);
	DeviceMemory<T>& operator=(const CudaMemory<T>&);

public:
	/**
	 * @brief コンストラクタ
	 *
	 * @param size [in] Tの個数
	 */
	DeviceMemory(size_t size) :
		_size(size) {
		// デバイスメモリの確保
		cudaMalloc((void**) &_device_mem, sizeof(T) * size);
	}

	/**
	 * @brief コンストラクタ
	 *
	 * @param size [in] Tの個数
	 * @param data [in] 初期化データ
	 * @param copy_to_device [in] デバイスにコピーするかどうか
	 */
	DeviceMemory(const T* data, size_t size) :
		_size(size) {

		// デバイスメモリの確保とコピー
		cudaMalloc((void**) &_device_mem, sizeof(T) * size);
		cudaMemcpy(_device_mem, data, sizeof(T) * size, cudaMemcpyHostToDevice);
	}

	virtual ~DeviceMemory() {
		if (_device_mem != NULL)
			cudaFree(_device_mem);
	}

	void fillZero() {
		cudaMemset(_device_mem, 0, sizeof(T) * _size);
	}

	/**
	 * @brief メモリサイズを返す
	 *
	 * Tの個数を取得する。バイト数はsizeof(T)*CudaMemory::size()。
	 *
	 * @return Tの個数
	 */
	size_t size() const {
		return _size;
	}

	void write(const T* data, size_t size) {
		cudaMemcpy(_device_mem, data, sizeof(T) * size, cudaMemcpyHostToDevice);
	}

	/**
	 * @brief 生データへのポインタ取得
	 */
	T* device_data() {
		return _device_mem;
	}
	const T* device_data() const {
		return _device_mem;
	}
};

#endif /* CUDAMEMORY_HPP_ */
