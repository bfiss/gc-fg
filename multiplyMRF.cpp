#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cassert>

using namespace std;

void loadMiddleburyMRFData(const char * filename, int* &data_positive, int* &data_negative, int* &hCue, int* &vCue, int &width, int &height, int &nLabels)
{
	FILE * fp;
	fp = fopen(filename,"rb");

	fscanf(fp,"%d %d %d",&width,&height,&nLabels);

	int i, n, x, y;
	int gt;
	for(i = 0; i < width * height; i++)
		fscanf(fp,"%d",&gt);

	assert(nLabels == 2);

	data_positive = (int*) malloc(width * height * sizeof(int));
	data_negative = (int*) malloc(width * height * sizeof(int));
	int v;

	for(i = 0; i < width * height; i++) {
		fscanf(fp,"%d",&v);
		data_positive[i] = v;
	}

	for(i = 0; i < width * height; i++) {
		fscanf(fp,"%d",&v);
		data_negative[i] = v;
	}

	hCue = (int*) malloc(width * height * sizeof(int));
	vCue = (int*) malloc(width * height * sizeof(int));

	for(y = 0; y < height; y++) {
		for(x = 0; x < width-1; x++) {
			fscanf(fp,"%d",&v);
			hCue[x+y*width] = v;
		}
	}

	for(y = 0; y < height-1; y++) {
		for(x = 0; x < width; x++) {
			fscanf(fp,"%d",&v);
			vCue[y*width+x] = v;
		}
	}
	for(x = 0; x < width; x++) {
		vCue[(height-1)*width+x] = 0;
		hCue[(height-1)*width+x] = 0;
	}

	fclose(fp);
}

void writeMiddleburyMRFData(int mult, int* data_positive, int* data_negative, int* hCue, int* vCue, int width, int height, int nLabels)
{
	printf("%d %d %d ",width*mult,height,nLabels);

	int i, n, x, y;
	int gt;
	for(i = 0; i < width*mult * height; i++)
		printf("%d ",1);

	for(i = 0 ; i < height ; ++i)
		for(int j = 0; j < width * mult; j++)
			printf("%d ",data_positive[i*width+(j%width)]);

	for(i = 0 ; i < height ; ++i)
		for(int j = 0; j < width * mult; j++)
			printf("%d ",data_negative[i*width+(j%width)]);

	for(y = 0 ; y < height ; y++) {
		for(x = 0 ; x < width * mult-1 ; x++) {
			printf("%d ",hCue[y*width+(x%(width-1))]);
		}
	}

	for(y = 0 ; y < height-1 ; y++) {
		for(x = 0 ; x < width * mult ; x++) {
			printf("%d ",vCue[y*width+(x%width)]);
		}
	}
}

void writeRihannaMRFData(int mult, int* data_positive, int* data_negative, int* hCue, int* vCue, int width, int height, int nLabels)
{
	printf("3600 3600 2 ");

	int i, n, x, y;
	int gt;
	for(i = 0; i < 3600*3600 ; i++)
		printf("1 ");

	for(i = 0 ; i < 3600 ; ++i)
		for(int j = 0; j < 3600 ; j++)
			printf("%d ",data_positive[i*width+(j%width)]);
	exit(1);

	for(i = 0 ; i < 3600 ; ++i)
		for(int j = 0; j < 3600 ; j++)
			printf("%d ",data_negative[i*width+(j%width)]);

	for(y = 0 ; y < 3600 ; y++) {
		for(x = 0 ; x < 3600-1 ; x++) {
			printf("600");
		}
	}

	for(y = 0 ; y < 3600-1 ; y++) {
		for(x = 0 ; x < 3600; x++) {
			printf("600");
		}
	}
}

int main(int argc, char * argv[]) {
	if(argc != 3) {
		printf("Usage: %s MDF_file number_of_multiplications",argv[0]);
		exit(1);
	}
	int* data_positive, * data_negative, * hCue, * vCue, width, height, nLabels;
	loadMiddleburyMRFData(argv[1],data_positive,data_negative,hCue,vCue,width,height,nLabels);

	/*ifstream face_file;
	face_file.open("rihanna.dat");
	//face_file.open("face.dat");

	face_file >> height >> width;

	data_positive = (int *) malloc(sizeof(int) * width * height);
	data_negative = (int *) malloc(sizeof(int) * width * height);
	for (int i = 0; i < height * width; ++i)
		face_file >> data_negative[i];
	for (int i = 0; i < height * width; ++i)
		face_file >> data_positive[i];
	face_file.close();*/

	int mult = atoi(argv[2]);

	//writeRihannaMRFData(mult,data_positive,data_negative,hCue,vCue,width,height,nLabels);
	writeMiddleburyMRFData(mult,data_positive,data_negative,hCue,vCue,width,height,nLabels);


	free(data_positive);
	free(data_negative);
	//free(hCue);
	//free(vCue);

	return 0;
}
