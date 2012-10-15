/*
 * util_cvlap.cpp
 *
 *  Created on: 2010/08/22
 *      Author: simasaki
 */

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "util_cv.h"

inline void SetUtilImageAttrFromplImage(UtilImage *_p_im, IplImage *_p_ipl_im) {
	(_p_im)->width = (_p_ipl_im)->width;
	(_p_im)->height = (_p_ipl_im)->height;
	(_p_im)->width_byte = (_p_ipl_im)->widthStep;
	(_p_im)->px_byte = (_p_ipl_im)->nChannels;
}

UtilCVImage *utilCV_CreateImage(int width, int height, int depth, int channels) {
	UtilCVImage *p_im = (UtilCVImage *) calloc(1, sizeof(UtilCVImage));
	IplImage *p_im_cv = cvCreateImage(cvSize(width, height), depth, channels);

	SetUtilImageAttrFromplImage(&(p_im->im), p_im_cv);
	p_im->im.p_buf = (void *) p_im_cv->imageData;
	p_im->p_iplimg = (void *) p_im_cv;
	return p_im;
}

UtilCVImage *utilCV_CloneImage(UtilCVImage *p_im_org) {
	UtilCVImage *p_im = (UtilCVImage *) calloc(1, sizeof(UtilCVImage));
	IplImage *p_im_cv = cvCloneImage((IplImage *) p_im_org->p_iplimg);

	SetUtilImageAttrFromplImage(&(p_im->im), p_im_cv);
	p_im->im.p_buf = (void *) p_im_cv->imageData;
	p_im->p_iplimg = (void *) p_im_cv;
	return p_im;
}

UtilCVImage *utilCV_LoadImage(const char *p_fname, UtilCVImageType im_type) {
	UtilCVImage *p_im = (UtilCVImage *) calloc(1, sizeof(UtilCVImage));
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

void utilCV_AddWeighted(UtilCVImage *p_im_src1, double alpha, UtilCVImage *p_im_src2, double beta, double gamma, UtilCVImage *p_im_dst) {
	cvAddWeighted((IplImage *) p_im_src1->p_iplimg, alpha, (IplImage *) p_im_src2->p_iplimg, beta, gamma,
		(IplImage *) p_im_dst->p_iplimg);
}

void utilCV_SaveImage(const char *p_fname, UtilCVImage *p_im) {
	IplImage *p_im_cv = (IplImage *) p_im->p_iplimg;
	cvSaveImage(p_fname, p_im_cv);
}

void utilCV_ReleaseImage(UtilCVImage **pp_im) {
	UtilCVImage *p_im = *pp_im;
	IplImage *p_im_cv = (IplImage *) p_im->p_iplimg;
	cvReleaseImage(&p_im_cv);
	free((void *) p_im);
	*pp_im = (UtilCVImage *) NULL;
}
