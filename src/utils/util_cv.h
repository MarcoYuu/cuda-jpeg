/*
 * util_cvlap.h
 *
 *  Created on: 2010/08/22
 *      Author: simasaki
 */

#ifndef UTIL_CVLAP_H_
#define UTIL_CVLAP_H_

#include "../type_definitions.h"
#include <string>

struct _IplImage;

class BitmapCVUtil{
private:
	_IplImage *_ipl_image;

public:
	enum ImageType {
		GRAYSCALE,
		RGB_COLOR
	};

public:
	BitmapCVUtil(const std::string &filename, ImageType type);
	BitmapCVUtil(int width, int height, int depth, int channels);
	~BitmapCVUtil();

	int getWidth() const;
	int getHeight() const;
	int getBytePerPixel() const;
	int getBytePerLine() const;

	void* getRawData();
	const void* getRawData() const;

	void saveToFile(const std::string &filename) const;

private:
	BitmapCVUtil(const BitmapCVUtil &rhs);
	BitmapCVUtil& operator=(const BitmapCVUtil &);
};

struct UtilImage {
	int width;
	int height;
	int px_byte;
	int width_byte;
	void *p_buf;
};

enum UtilCVImageType {
	UTIL_CVIM_GRAYSCALE = 0,
	UTIL_CVIM_COLOR = 1
};

struct UtilCVImageStruct {
	UtilImage im;
	void *p_iplimg;
};

UtilCVImageStruct *utilCV_CreateImage(int width, int height, int depth, int channels);
UtilCVImageStruct *utilCV_CloneImage(UtilCVImageStruct *p_im_org);
UtilCVImageStruct *utilCV_LoadImage(const char *p_fname, UtilCVImageType im_type);

void utilCV_AddWeighted(UtilCVImageStruct *p_im_src1, double alpha, UtilCVImageStruct *p_im_src2, double beta, double gamma, UtilCVImageStruct *p_im_dst);

void utilCV_SaveImage(const char *p_fname, UtilCVImageStruct *p_im);
void utilCV_ReleaseImage(UtilCVImageStruct **pp_im);

#endif /* UTIL_CVLAP_H_ */
