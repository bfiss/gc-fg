#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include "GraphCut.cu"

#include <string.h>
#include <stdio.h>

#include "imageIO.h"
//#include "DataTerm.h"
#include "DataTerm.cpp"
using namespace std;

#define USE_CPU false
#define MIXTURE false
#define VIDEO true
#define BATCHMODE false

uchar* out_data;
uchar* inp_data;
int* data_pos;
int* data_neg;
int* labels;

IplImage* Image;
CvCapture* input_video;

int width = 512;
int height = 512;
float  hue, win;
int penalty;
bool stopped;


GlobalWrapper gw;
void process();
void resetSelection();
void makeSelection(int, int);
void drawSelection(uchar*);
void nextFrame();
void loadVideo(char*);

void penTrackbarCallback(int val)
{
	penalty = val*10;
	printf("pen = %d, hue = %1.2f, win = %1.2f\n", penalty, hue, win);

	process();
	displayImage(out_data, "win1");
}

void hueTrackbarCallback(int val)
{
	hue = val/4.0;
	printf("pen = %d, hue = %1.2f, win = %1.2f\n", penalty, hue, win);

	process();
}

void winTrackbarCallback(int val)
{
	win = val;
	printf("pen = %d, hue = %1.2f, win = %1.2f\n", penalty, hue, win);

	process();
}

void mouseCallback (int event, int x, int y, int flags, void* param)
{
	switch(event){

		case CV_EVENT_RBUTTONUP:

			DataTerm::setImage(inp_data);
			DataTerm::getDataTermsMixture(data_pos, data_neg);
			//DataTerm::drawSelection(inp_data, out_data);
			process();
			displayImage(out_data, "win1");

	        break;

		case CV_EVENT_MBUTTONUP:

			DataTerm::reset();
			displayImage(out_data, "win1");

		case CV_EVENT_LBUTTONUP:

			printf("%d %d\n", x,y);
			DataTerm::selectOnMap(x, y, 8);
	        DataTerm::drawSelection(inp_data, out_data);
	    	displayImage(out_data, "win1");

	    	break;
	}
}

// init client side data structures [todo: move to device]
// init graph cut stuff
// init data terms stuff
// call after loadImage, that we know width & height
void init()
{
	gw = GC_Init(width, height);	// set penalty and dataterms later
	DataTerm::init_data_term(width, height);

	labels = (int *) malloc(sizeof(int) * width * height);

	// alloc data terms
	data_pos = (int*)malloc(sizeof(int)*width*height);
	data_neg = (int*)malloc(sizeof(int)*width*height);
}

void stop()
{
	DataTerm::free_data_term();
	GC_End(&gw);

	free(data_pos);
	free(data_neg);
	free(out_data);
	free(inp_data);
	free(labels);
}

void process()
{
	GC_SetDataterms(&gw, DataTerm::getPos(), DataTerm::getNeg());
	GC_SetPenalty(&gw, penalty);
	GC_SetGraph(gw);
	GC_Optimize(gw, labels);
	DataTerm::setSelection(labels);
	DataTerm::drawSelection(inp_data, out_data);
}

void loadImage(char* name)
{
	Image = cvLoadImage(name);
	width  = Image->width;
	height = Image->height;


	initDisplay(width, height);

	// copy image data to input array
	out_data = (uchar*)malloc(sizeof(uchar)*width*height*3);
	inp_data = (uchar*)malloc(sizeof(uchar)*width*height*3);
	memcpy(inp_data, Image->imageData, sizeof(uchar)*width*height*3);
	memcpy(out_data, Image->imageData, sizeof(uchar)*width*height*3);
}

void loadVideo(char* name)
{
	input_video = cvCaptureFromFile(name); //cvCaptureFromAVI cvCaptureFromFile(name);
	Image = 0;

	if(!cvGrabFrame(input_video))
	{
	    printf("Could not grab a frame\n\7");
	    exit(0);
	}
	Image = cvRetrieveFrame(input_video);
	width = Image->width;
	height = Image->height;

	initDisplay(width, height);
	out_data = (uchar*)malloc(sizeof(uchar)*width*height*3);
	inp_data = (uchar*)malloc(sizeof(uchar)*width*height*3);
	memcpy(inp_data, Image->imageData, sizeof(uchar)*width*height*3);
	memcpy(out_data, Image->imageData, sizeof(uchar)*width*height*3);
}

void nextFrame()
{
	if(!cvGrabFrame(input_video))
	{
		printf("Could not grab a frame\n\7");
		exit(0);
	}
	Image = cvRetrieveFrame(input_video);
	memcpy(inp_data, Image->imageData, sizeof(uchar)*width*height*3);
	memcpy(out_data, Image->imageData, sizeof(uchar)*width*height*3);
}


int main()
{
	stopped = false;
	hue = 0;
	win = 0;

#if VIDEO
	loadVideo("golf.mp4");
#else
	loadImage("group.jpg");
#endif

	init();

#if BATCHMODE

	penalty = 7000;
	DataTerm::setImage(inp_data);
	DataTerm::selectOnMap(200, 70, 8);
	DataTerm::selectOnMap(204, 62, 8);
	DataTerm::selectOnMap(201, 68, 8);

	DataTerm::getDataTermsMixture(data_pos, data_neg);

	process();

	//displayImage(out_data, "win1");


#else
	char key;
	while(1){

#if VIDEO
		nextFrame();
		DataTerm::setImage(out_data);
		DataTerm::getDataTermsMixture(data_pos, data_neg);
		process();
#endif
		displayImage(out_data, "win1");

		// play / pause on space
		if (stopped || !VIDEO)
			key=cvWaitKey(0);
		else
			key=cvWaitKey(2);

		// quit on esc
		if(key==27) break;

		// optional input
	    switch(key){
	    	case 32:
	    		stopped ^= true;
	    		break;
	    	case 49:
	    		printf("load group");
	    		loadImage("group.jpg");
		        break;
	    	case 50:
	    		printf("load hue");
	    		loadImage("hue.jpg");
	    		break;
	    	default:
	    		break;
	    }
	}
#endif

	//cvWaitKey();
	stop();
	return 0;
}
