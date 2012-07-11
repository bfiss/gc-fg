/*! @file
 * imageIO.h
 * Functions and data structures for image and video loading. Basic gui element creation and image display.
 *
 *  Created on: Jun 13, 2012
 *      Author: max
 */

#ifndef IMAGEIO_H_
#define IMAGEIO_H_
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"


//!Stores all image related data. Is used for file input and display.
struct ImageWrapper
{
	int width; 				 //!< width of current loaded image / video
	int height;				 //!< height of current loaded image / video
	IplImage* InputImage;    //!< input image, loaded by loadImage()
	IplImage* DisplayImage;  //!< image structure for display output.
	CvCapture* InputVideo;   //!< capture structure and framesource for input video,l loaded by loadVideo().
	unsigned char* inp_data; //!< raw image data retrieved from input image / frame. Data is interleaved BGRBGR...
	unsigned char* out_data; //!< raw image data that is displayed when displayImage() is called. Must be interleaved BGRBGR...
};

void winTrackbarCallback(int val);
void penTrackbarCallback(int val);
void mouseCallback(int event, int x, int y, int flags, void* param);


void initDisplay(ImageWrapper* imw)
{
	/*!
	 * Create open cv windows and trackbars. Set callback functions for mouse and trackbar action.
	 * @param[in] imw image wrapper.
	 */
	CvSize s;
	s.height = imw->height;
	s.width  = imw->width;

	int val0;
	int val1;
	int val2;
	cvNamedWindow("win_control", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("win1", CV_WINDOW_AUTOSIZE);
	imw->DisplayImage = cvCreateImage( s, IPL_DEPTH_8U, 3);

	cvCreateTrackbar("Pen", "win_control", &val0, 1000, penTrackbarCallback);
	cvCreateTrackbar("Win", "win_control", &val2, 20, winTrackbarCallback);

	int mouseParam = 5;
	cvSetMouseCallback("win1",mouseCallback,&mouseParam);
}

void displayImage(char* windowName, ImageWrapper* imw)
{
	/*!
	 * Display image in window.
	 * @param[in] windowName name of the display window, type cvNamedWindow
	 * @param[in] imw Image wrapper, that contains, output raw image data to be displayed.
	 */
	memcpy(imw->DisplayImage->imageData, imw->out_data, imw->DisplayImage->imageSize);
	cvShowImage(windowName, imw->DisplayImage);
}

void loadImage(char* name, ImageWrapper* imw)
{
	/*!
	 * Load image from file.
	 * @param[in] name name of the image file. Supported types: jpg, bmp, gif and others.
	 * @param[in] imw Image wrapper. Raw data is copied to imw->inp_data. Image resolution is stored in imw->width, imw->height.
	 */
	imw->InputImage = cvLoadImage(name);
	imw->width  = imw->InputImage->width;
	imw->height = imw->InputImage->height;

	initDisplay(imw);
	// copy image data to input array
	imw->out_data = (uchar*)malloc(sizeof(uchar)*imw->width*imw->height*3);
	imw->inp_data = (uchar*)malloc(sizeof(uchar)*imw->width*imw->height*3);
	memcpy(imw->inp_data, imw->InputImage->imageData, sizeof(uchar)*imw->width*imw->height*3);
	memcpy(imw->out_data, imw->InputImage->imageData, sizeof(uchar)*imw->width*imw->height*3);
}

void loadVideo(char* name, ImageWrapper* imw)
{
	/*!
	 * Load video from file and copy first frame to imw->inp_data.
	 * Supported container types: avi, mp4. Supported codecs mpeg4. See opencv documentation.
	 * @param[in] name name of the video file.
	 * @param[in] imw Image wrapper. Raw data of current frame is copied to imw->inp_data. Video resolution is stored in imw->width, imw->height.
	 */
	imw->InputVideo = cvCaptureFromFile(name); //cvCaptureFromAVI cvCaptureFromFile(name);
	imw->InputImage = 0;

	if(!cvGrabFrame(imw->InputVideo))
	{
	    printf("Could not grab a frame\n\7");
	    exit(0);
	}
	imw->InputImage = cvRetrieveFrame(imw->InputVideo);
	imw->width = imw->InputImage->width;
	imw->height = imw->InputImage->height;

	initDisplay(imw);
	imw->out_data = (uchar*)malloc(sizeof(uchar)*imw->width*imw->height*3);
	imw->inp_data = (uchar*)malloc(sizeof(uchar)*imw->width*imw->height*3);
	memcpy(imw->inp_data, imw->InputImage->imageData, sizeof(uchar)*imw->width*imw->height*3);
	memcpy(imw->out_data, imw->InputImage->imageData, sizeof(uchar)*imw->width*imw->height*3);
}

void seekFrame(ImageWrapper* imw, int pos)
{
	/*!
	 * Seek to frame position in input video.
	 * @param[in] imw image wrapper.
	 * @param[in] pos number of frame to be seeked.
	 */
	cvSetCaptureProperty( imw->InputVideo, CV_CAP_PROP_POS_FRAMES , pos );
}

void nextFrame(ImageWrapper* imw)
{
	/*!
	 * Get next frame from input video.
	 * @param[in] imw Image wrapper. Frame raw data is stored in imw->inp_data.
	 */

	if(!cvGrabFrame(imw->InputVideo))
	{
		printf("Could not grab a frame\n\7");
		exit(0);
	}
	imw->InputImage = cvRetrieveFrame(imw->InputVideo);
	memcpy(imw->inp_data, imw->InputImage->imageData, sizeof(uchar)*imw->width*imw->height*3);
	memcpy(imw->out_data, imw->InputImage->imageData, sizeof(uchar)*imw->width*imw->height*3);
}


#endif /* IMAGEIO_H_ */
