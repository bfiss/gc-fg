#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include "GraphCut.cu"

using namespace std;

#define HEIGHT 400
#define WIDTH 400

#define SIZE (HEIGHT*WIDTH)

int main(int argc, char * argv[]) {
	int width;
	int height;

	/*height = 5;
	width = 5;

	int data_positive[] = {0, 0, 0, 0, 0, 0, 6, 4, 3, 0, 0, 5, 3, 2, 0, 0, 4, 2, 1, 0, 0, 0, 0, 0, 0};
	int data_negative[] = {0, 0, 0, 0, 0, 0, 3, 4, 3, 0, 0, 3, 4, 3, 0, 0, 3, 4, 4, 0, 0, 0, 0, 0, 0};*/

	height = 15;
	width = 4;

	int data_positive[SIZE] = { 0, 0, 0, 0,
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
	int data_negative[SIZE] = { 0, 0, 0, 0,
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

	if(argc == 2) {
		srand(atoi(argv[1]));
		height = HEIGHT;
		width = WIDTH;
		for(int i = 0 ; i < height ; ++i)
			for(int j = 0 ; j < width ; ++j) {
				if(false /*|| i == 0 || j == 0 || i == height-1 || j == width-1*/) {
					data_positive[i*width+j] = 0;
					data_negative[i*width+j] = 0;
				} else {
					data_positive[i*width+j] = rand()%300;
					data_negative[i*width+j] = rand()%300;
				}
			}
	}

	int * d_data_positive, * d_data_negative;

	CUDA_SAFE_CALL(cudaMalloc((void**)&(d_data_positive),sizeof(int)*width*height));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(d_data_negative),sizeof(int)*width*height));
	CUDA_SAFE_CALL(cudaMemcpy(d_data_positive,data_positive,sizeof(int)*width*height,cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_data_negative,data_negative,sizeof(int)*width*height,cudaMemcpyHostToDevice));

	GlobalWrapper gw =
			GC_Init(width, height, d_data_positive, d_data_negative, 50);

	int * label = (int *) malloc(sizeof(int) * width * height);

	GC_Optimize(gw, label);

	/*for(int i = 0 ; i < height ; ++i) {
		for(int j = 0 ; j < width ; ++j)
	 	 	 cout << label[i*width+j] << " ";
	 	 cout << endl;
	}*/

	free(label);

	CUDA_SAFE_CALL(cudaFree(d_data_positive));
	CUDA_SAFE_CALL(cudaFree(d_data_negative));

	GC_End(&gw);

	return 0;
}

