/*
 * imageIO.h
 *
 *  Created on: Jun 13, 2012
 *      Author: max
 */

#ifndef IMAGEIO_H_
#define IMAGEIO_H_
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

/*
struct Img{
	uchar* data;
	int width;
	int height;
};*/

IplImage* Img;

void hueTrackbarCallback(int val);
void winTrackbarCallback(int val);
void penTrackbarCallback(int val);
void mouseCallback(int event, int x, int y, int flags, void* param);

void initDisplay(int width, int height)
{
	CvSize s;
	s.height = height;
	s.width = width;

	int val0;
	int val1;
	int val2;
	cvNamedWindow("win_control", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("win1", CV_WINDOW_AUTOSIZE);
	Img = cvCreateImage( s, IPL_DEPTH_8U, 3);

	cvCreateTrackbar("Pen", "win_control", &val0, 1000, penTrackbarCallback);
	cvCreateTrackbar("Hue", "win_control", &val1, 360*4, hueTrackbarCallback);
	cvCreateTrackbar("Win", "win_control", &val2, 100, winTrackbarCallback);

	int mouseParam = 5;
	cvSetMouseCallback("win1",mouseCallback,&mouseParam);
}

void displayImage(uchar* data, char* windowName)
{
	printf("display");
	memcpy(Img->imageData, data, Img->imageSize);
	printf("mem");
	cvShowImage(windowName, Img);
}



/*
void loadImage(Img* inp, char* name)
{
	IplImage* cvimg = cvLoadImage(name);
	int size = cvimg->imageSize;
	inp->data = (uchar*)malloc(size);
	memcpy(inp->data, cvimg->imageData, size);
}*/




#endif /* IMAGEIO_H_ */
