#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include "GraphCut.cu"

using namespace std;

int main() {
	int width;
	int height;

	/*height = 5;
	width = 5;

	int data_positive[] = {0, 0, 0, 0, 0, 0, 6, 4, 3, 0, 0, 5, 3, 2, 0, 0, 4, 2, 1, 0, 0, 0, 0, 0, 0};
	int data_negative[] = {0, 0, 0, 0, 0, 0, 3, 4, 3, 0, 0, 3, 4, 3, 0, 0, 3, 4, 4, 0, 0, 0, 0, 0, 0};*/

	height = 15;
	width = 4;

	int data_positive[] = { 0, 0, 0, 0,
							0, 4, 0, 0,
							0, 0, 2, 0,
							0, 0, 0, 0,
							0, 0, 0, 0,
							0, 0, 0, 0,
							0, 0, 0, 0,
							0, 0, 0, 0,
							0, 0, 0, 0,
							0, 0, 0, 0,
							0, 0, 0, 0,
							0, 0, 0, 0,
							0, 0, 0, 0,
							0, 1, 0, 0,
							0, 0, 0, 0
	};
	int data_negative[] = { 0, 0, 0, 0,
							0, 0, 1, 0,
							0, 2, 0, 0,
							0, 0, 0, 0,
							0, 0, 0, 0,
							0, 0, 0, 0,
							0, 0, 0, 0,
							0, 0, 0, 0,
							0, 0, 0, 0,
							0, 0, 0, 0,
							0, 0, 0, 0,
							0, 0, 1, 0,
							0, 0, 0, 0,
							0, 0, 4, 0,
							0, 0, 0, 0
	};

	srand(time(NULL));

	int * d_data_positive, * d_data_negative;

	CUDA_SAFE_CALL(cudaMalloc((void**)&(d_data_positive),sizeof(int)*width*height));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(d_data_negative),sizeof(int)*width*height));
	CUDA_SAFE_CALL(cudaMemcpy(d_data_positive,data_positive,sizeof(int)*width*height,cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_data_negative,data_negative,sizeof(int)*width*height,cudaMemcpyHostToDevice));

	GlobalWrapper gw =
			GC_Init(width, height, d_data_positive, d_data_negative, 2);

	int * label = (int *) malloc(sizeof(int) * width * height);

	GC_Optimize(gw, label);

	for(int i = 0 ; i < height ; ++i) {
		for(int j = 0 ; j < width ; ++j)
	 	 	 cout << label[i*width+j] << " ";
	 	 cout << endl;
	}

	free(label);

	CUDA_SAFE_CALL(cudaFree(d_data_positive));
	CUDA_SAFE_CALL(cudaFree(d_data_negative));

	GC_End(&gw);

	return 0;
}

