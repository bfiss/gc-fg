#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include "GraphCut.cu"

using namespace std;

int main() {
	int width;
	int height;

	ifstream face_file;
	face_file.open("rihanna.dat");
	//face_file.open("face.dat");

	face_file >> height >> width;

	int * data_positive = (int *) malloc(sizeof(int) * width * height);
	int * data_negative = (int *) malloc(sizeof(int) * width * height);
	for (int i = 0; i < height * width; ++i)
		face_file >> data_negative[i];
	for (int i = 0; i < height * width; ++i)
		face_file >> data_positive[i];
	face_file.close();

	int * d_data_positive, * d_data_negative;

	CUDA_SAFE_CALL(cudaMalloc((void**)&(d_data_positive),sizeof(int)*width*height));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(d_data_negative),sizeof(int)*width*height));
	CUDA_SAFE_CALL(cudaMemcpy(d_data_positive,data_positive,sizeof(int)*width*height,cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_data_negative,data_negative,sizeof(int)*width*height,cudaMemcpyHostToDevice));

	srand( time(NULL));

	GlobalWrapper gw =
			GC_Init(width, height, d_data_positive, d_data_negative, 600);
	free(data_positive);
	free(data_negative);

	int * label = (int *) malloc(sizeof(int) * width * height);

	GC_Optimize(gw, label);

	/*
	for(int i = 0 ; i < height ; ++i) {
		for(int j = 0 ; j < width ; ++j)
	 	 	 cout << label[i*width+j] << " ";
	 	 cout << endl;
	}
	*/

	// read image
	ifstream face_in;
	int dumb_i;
	string dumb_s;
	face_in.open("rihanna.ppm");
	//face_in.open("face.ppm");
	face_in >> dumb_s >> dumb_i >> dumb_i >> dumb_i;
	
	// print processed image
	ofstream face_out;
	face_out.open("label.ppm");
	face_out << "P3 " << width << " " << height << " 255 " << endl;
	for (unsigned i = 0; i < height; i++) {
		for (unsigned j = 0; j < width; j++) {
			int c_a, c_b, c_c;
			face_in >> c_a >> c_b >> c_c;
			if (label[i * width + j]) {
				face_out << c_a << " " << c_b << " " << c_c << " ";
			} else {
				face_out << 0 << " " << 0 << " " << 0 << " ";
			}
		}
		face_out << endl;
	}
	face_in.close();
	face_out.close();
	free(label);

	CUDA_SAFE_CALL(cudaFree(d_data_positive));
	CUDA_SAFE_CALL(cudaFree(d_data_negative));

	GC_End(&gw);

	return 0;
}

