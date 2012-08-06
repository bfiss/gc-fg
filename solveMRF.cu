#include <iostream>
#include <ctime>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cassert>

using namespace std;

void loadMiddleburyMRFData(const char * filename, int* &data_positive, int* &data_negative, int* &hCue, int* &vCue, int &width, int &height, int &nLabels)
{
	FILE * fp;
	fp = fopen(filename,"rb");
	assert(fp);

	assert(fscanf(fp,"%d %d %d",&width,&height,&nLabels)==3);

	int i, gt;
	for(i = 0; i < width * height; i++)
		assert(fscanf(fp,"%d",&gt)==1);
	assert(gt == gt);

	assert(nLabels == 2);

	data_positive = (int*) malloc(width * height * sizeof(int));
	data_negative = (int*) malloc(width * height * sizeof(int));
	assert(data_positive && data_negative);

	int v;

	for(i = 0; i < width * height; i++) {
		assert(fscanf(fp,"%d",&v)==1);
		data_positive[i] = v;
	}

	for(i = 0; i < width * height; i++) {
		assert(fscanf(fp,"%d",&v)==1);
		data_negative[i] = v;
	}

	hCue = (int*) malloc(width * height * sizeof(int));
	vCue = (int*) malloc(width * height * sizeof(int));
	assert(hCue && vCue);

	int x, y;
	for(y = 0; y < height; y++) {
		for(x = 0; x < width-1; x++) {
			assert(fscanf(fp,"%d",&v)==1);
			hCue[x+y*width] = v;
		}
	}

	for(y = 0; y < height-1; y++) {
		for(x = 0; x < width; x++) {
			assert(fscanf(fp,"%d",&v)==1);
			vCue[y*width+x] = v;
		}
	}
	for(x = 0; x < width; x++) {
		vCue[(height-1)*width+x] = 0;
		hCue[(height-1)*width+x] = 0;
	}

	fclose(fp);
}

#include "GraphCut.cu"

int main(int argc, char * argv[]) {
	if(argc != 2) {
		printf("Usage: %s MDF_file\n",argv[0]);
		exit(1);
	}
	int* data_positive, * data_negative, * hCue, * vCue, width, height, nLabels;
	loadMiddleburyMRFData(argv[1],data_positive,data_negative,hCue,vCue,width,height,nLabels);

	int * d_data_positive, * d_data_negative, * d_up, * d_down, * d_left, * d_right;

	CUDA_SAFE_CALL(cudaMalloc((void**)&(d_data_positive),sizeof(int)*width*height));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(d_data_negative),sizeof(int)*width*height));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(d_up),sizeof(int)*width*height));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(d_down),sizeof(int)*width*height));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(d_left),sizeof(int)*width*height));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(d_right),sizeof(int)*width*height));
	CUDA_SAFE_CALL(cudaMemcpy(d_data_positive,data_positive,sizeof(int)*width*height,cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_data_negative,data_negative,sizeof(int)*width*height,cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_up,vCue,sizeof(int)*width*height,cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_down,vCue,sizeof(int)*width*height,cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_left,hCue,sizeof(int)*width*height,cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_right,hCue,sizeof(int)*width*height,cudaMemcpyHostToDevice));

	free(data_positive);
	free(data_negative);
	free(hCue);
	free(vCue);

	srand( time(NULL));

	printf("Solving a %d x %d MRF problem...\n",height,width);

	if(NEIGHBORHOOD != 4)
		printf("Please change NEIGHBORHOOD to 4\n");
	assert(NEIGHBORHOOD == 4);

	GlobalWrapper gw =
			GC_Init(width, height, d_data_positive, d_data_negative, 0, d_up, d_down, d_left, d_right);

	int * label = (int *) malloc(sizeof(int) * width * height);
	assert(label);

	GC_Optimize(gw, label);

	// print processed image
	ofstream face_out;
	face_out.open("labelMRF.ppm");
	face_out << "P3 " << width << " " << height << " 255 " << endl;
	for (unsigned i = 0; i < height; i++) {
		for (unsigned j = 0; j < width; j++) {
			if (label[i * width + j]) {
				face_out << 255 << " " << 255 << " " << 255 << " ";
			} else {
				face_out << 0 << " " << 0 << " " << 0 << " ";
			}
		}
		face_out << endl;
	}
	face_out.close();

	free(label);

	CUDA_SAFE_CALL(cudaFree(d_data_positive));
	CUDA_SAFE_CALL(cudaFree(d_data_negative));
	CUDA_SAFE_CALL(cudaFree(d_up));
	CUDA_SAFE_CALL(cudaFree(d_down));
	CUDA_SAFE_CALL(cudaFree(d_left));
	CUDA_SAFE_CALL(cudaFree(d_right));

	GC_End(&gw);

	return 0;
}
