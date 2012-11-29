/*
 * util_cvlap.cpp
 *
 *  Created on: 2010/08/22
 *      Author: simasaki
 */

#include <algorithm>

#define _CRT_SECURE_NO_DEPRECATE

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "util_cv.h"

namespace util {
	inline void SetUtilImageAttrFromplImage(UtilImage *_p_im, IplImage *_p_ipl_im) {
		(_p_im)->width = (_p_ipl_im)->width;
		(_p_im)->height = (_p_ipl_im)->height;
		(_p_im)->width_byte = (_p_ipl_im)->widthStep;
		(_p_im)->px_byte = (_p_ipl_im)->nChannels;
	}

	UtilCVImageStruct *utilCV_CreateImage(int width, int height, int depth,
		int channels) {
		UtilCVImageStruct *p_im = (UtilCVImageStruct *) calloc(1,
			sizeof(UtilCVImageStruct));
		IplImage *p_im_cv = cvCreateImage(cvSize(width, height), depth, channels);

		SetUtilImageAttrFromplImage(&(p_im->im), p_im_cv);
		p_im->im.p_buf = (void *) p_im_cv->imageData;
		p_im->p_iplimg = (void *) p_im_cv;
		return p_im;
	}

	UtilCVImageStruct *utilCV_CloneImage(UtilCVImageStruct *p_im_org) {
		UtilCVImageStruct *p_im = (UtilCVImageStruct *) calloc(1,
			sizeof(UtilCVImageStruct));
		IplImage *p_im_cv = cvCloneImage((IplImage *) p_im_org->p_iplimg);

		SetUtilImageAttrFromplImage(&(p_im->im), p_im_cv);
		p_im->im.p_buf = (void *) p_im_cv->imageData;
		p_im->p_iplimg = (void *) p_im_cv;
		return p_im;
	}

	UtilCVImageStruct *utilCV_LoadImage(const char *p_fname, UtilCVImageType im_type) {
		UtilCVImageStruct *p_im = (UtilCVImageStruct *) calloc(1,
			sizeof(UtilCVImageStruct));
		IplImage *p_im_cv;

		switch (im_type) {
		case UTIL_CVIM_GRAYSCALE:
			p_im_cv = cvLoadImage(p_fname, CV_LOAD_IMAGE_GRAYSCALE);
			break;

		case UTIL_CVIM_COLOR:
		default:
			p_im_cv = cvLoadImage(p_fname, CV_LOAD_IMAGE_COLOR);
			break;
		}

		SetUtilImageAttrFromplImage(&(p_im->im), p_im_cv);
		p_im->im.p_buf = (void *) p_im_cv->imageData;
		p_im->p_iplimg = (void *) p_im_cv;
		return p_im;
	}

	void utilCV_AddWeighted(UtilCVImageStruct *p_im_src1, double alpha,
		UtilCVImageStruct *p_im_src2, double beta, double gamma,
		UtilCVImageStruct *p_im_dst) {
		cvAddWeighted((IplImage *) p_im_src1->p_iplimg, alpha,
			(IplImage *) p_im_src2->p_iplimg, beta, gamma,
			(IplImage *) p_im_dst->p_iplimg);
	}

	void utilCV_SaveImage(const char *p_fname, UtilCVImageStruct *p_im) {
		IplImage *p_im_cv = (IplImage *) p_im->p_iplimg;
		cvSaveImage(p_fname, p_im_cv);
	}

	void utilCV_ReleaseImage(UtilCVImageStruct **pp_im) {
		UtilCVImageStruct *p_im = *pp_im;
		IplImage *p_im_cv = (IplImage *) p_im->p_iplimg;
		cvReleaseImage(&p_im_cv);
		free((void *) p_im);
		*pp_im = (UtilCVImageStruct *) NULL;
	}

	BitmapCVUtil::BitmapCVUtil(const std::string &filename, ImageType type) {
		switch (type) {
		case UTIL_CVIM_GRAYSCALE:
			_ipl_image = cvLoadImage(filename.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
			break;

		case UTIL_CVIM_COLOR:
		default:
			_ipl_image = cvLoadImage(filename.c_str(), CV_LOAD_IMAGE_COLOR);
			break;
		}
	}

	BitmapCVUtil::BitmapCVUtil(int width, int height, int depth, int channels) {
		_ipl_image = cvCreateImage(cvSize(width, height), depth, channels);
	}

	BitmapCVUtil::~BitmapCVUtil() {
		cvReleaseImage(&_ipl_image);
	}

	int BitmapCVUtil::getWidth() const {
		return _ipl_image->width;
	}

	int BitmapCVUtil::getHeight() const {
		return _ipl_image->height;
	}

	int BitmapCVUtil::getBytePerPixel() const {
		return _ipl_image->nChannels;
	}

	int BitmapCVUtil::getBytePerLine() const {
		return _ipl_image->widthStep;
	}

	void* BitmapCVUtil::getRawData() {
		return _ipl_image->imageData;
	}

	const void* BitmapCVUtil::getRawData() const {
		return _ipl_image->imageData;
	}

	void BitmapCVUtil::saveToFile(const std::string &filename) const {
		cvSaveImage(filename.c_str(), _ipl_image);
	}
} // namespace util
