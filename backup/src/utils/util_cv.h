/*
 * util_cvlap.h
 *
 *  Created on: 2010/08/22
 *      Author: simasaki
 */

#ifndef UTIL_CVLAP_H_
#define UTIL_CVLAP_H_

//#include "util.h"
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

struct UtilCVImage {
	UtilImage im;
	void *p_iplimg;
};

UtilCVImage *utilCV_CreateImage(int width, int height, int depth, int channels);
UtilCVImage *utilCV_CloneImage(UtilCVImage *p_im_org);
UtilCVImage *utilCV_LoadImage(const char *p_fname, UtilCVImageType im_type);

void utilCV_AddWeighted(UtilCVImage *p_im_src1, double alpha, UtilCVImage *p_im_src2, double beta, double gamma, UtilCVImage *p_im_dst);

void utilCV_SaveImage(const char *p_fname, UtilCVImage *p_im);
void utilCV_ReleaseImage(UtilCVImage **pp_im);

#endif /* UTIL_CVLAP_H_ */
