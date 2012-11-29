/*
 * util_cvlap.h
 *
 *  Created on: 2010/08/22
 *      Author: simasaki
 */

#ifndef UTIL_CVLAP_H_
#define UTIL_CVLAP_H_

#include "type_definitions.h"
#include <string>

struct _IplImage;

namespace util {
	/**
	 * @brief Bitmap読み書きに限定したOpenCVヘルパー
	 *
	 * @author yuumomma
	 * @version 1.0
	 */
	class BitmapCVUtil {
	private:
		_IplImage *_ipl_image;

	public:
		enum ImageType {
			GRAYSCALE, RGB_COLOR
		};

	public:
		/**
		 * @brief コンストラクタ
		 *
		 * ファイルから作成
		 *
		 * @param filename ファイル名
		 * @param type タイプ
		 */
		BitmapCVUtil(const std::string &filename, ImageType type);
		/**
		 * @brief コンストラクタ
		 *
		 * 新規作成
		 *
		 * @param width 幅
		 * @param height 高さ
		 * @param depth 色深度
		 * @param channels 色数
		 */
		BitmapCVUtil(int width, int height, int depth, int channels);
		/**
		 * @brief デストラクタ
		 */
		~BitmapCVUtil();

		/**
		 * @brief 幅取得
		 *
		 * @return 幅
		 */
		int getWidth() const;
		/**
		 * @brief 高さ取得
		 *
		 * @return 高さ
		 */
		int getHeight() const;
		/**
		 * @brief ピクセル毎のバイト数取得
		 *
		 * @return バイト数
		 */
		int getBytePerPixel() const;
		/**
		 * @brief 一行あたりのバイト数取得
		 *
		 * @return バイト数
		 */
		int getBytePerLine() const;

		/**
		 * @brief 生ビットマップデータへのポインタ取得
		 */
		void* getRawData();
		/**
		 * @brief 生ビットマップデータへのポインタ取得
		 */
		const void* getRawData() const;

		/**
		 * @brief 画像をファイルに保存
		 *
		 * 形式は拡張子に依存.「.bmp」で終わる必要がある
		 *
		 * @param filename ファイル名
		 */
		void saveToFile(const std::string &filename) const;

	private:
		BitmapCVUtil(const BitmapCVUtil &rhs);
		BitmapCVUtil& operator=(const BitmapCVUtil &);
	};

	/**
	 * @brief CV画像ヘルパ構造体
	 *
	 * @deprecated
	 * @author ohmura
	 * @version 1.0
	 */
	struct UtilImage {
		int width;
		int height;
		int px_byte;
		int width_byte;
		void *p_buf;
	};

	/**
	 * @brief 画像タイプ
	 *
	 * @deprecated
	 * @author ohmura
	 * @version 1.0
	 */
	enum UtilCVImageType {
		UTIL_CVIM_GRAYSCALE = 0, UTIL_CVIM_COLOR = 1
	};

	/**
	 * @brief CV画像ヘルパ構造体
	 *
	 * @deprecated
	 * @author ohmura
	 * @version 1.0
	 */
	struct UtilCVImageStruct {
		UtilImage im;
		void *p_iplimg;
	};

	/**
	 * @brief OpenCVを利用してイメージを作成する
	 *
	 * @deprecated
	 * @param width 幅
	 * @param height 高さ
	 * @param depth 色深度
	 * @param channels 色数
	 * @return イメージ
	 */
	UtilCVImageStruct *utilCV_CreateImage(int width, int height, int depth, int channels);
	/**
	 * @brief 複製する
	 *
	 * @deprecated
	 * @param p_im_org 複製元
	 * @return 複製イメージ
	 */
	UtilCVImageStruct *utilCV_CloneImage(UtilCVImageStruct *p_im_org);
	/**
	 * @brief ファイルからOpenCVのイメージを作成する
	 *
	 * @deprecated
	 * @param p_fname ファイル名
	 * @param im_type 画像タイプ
	 * @sa UtilCVImageType
	 * @return イメージ
	 */
	UtilCVImageStruct *utilCV_LoadImage(const char *p_fname, UtilCVImageType im_type);

	void utilCV_AddWeighted(UtilCVImageStruct *p_im_src1, double alpha, UtilCVImageStruct *p_im_src2,
		double beta, double gamma, UtilCVImageStruct *p_im_dst);

	/**
	 * @brief イメージを保存する
	 *
	 * @deprecated
	 * @param p_fname 出力ファイル名、形式は拡張子に依存
	 * @param p_im 保存するイメージ
	 */
	void utilCV_SaveImage(const char *p_fname, UtilCVImageStruct *p_im);
	/**
	 * @brief イメージのメモリを開放する
	 *
	 * @deprecated
	 * @param pp_im イメージ
	 */
	void utilCV_ReleaseImage(UtilCVImageStruct **pp_im);
} // namespace util

#endif /* UTIL_CVLAP_H_ */
