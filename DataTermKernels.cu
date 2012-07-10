/*
 * DataTermKernels.cu
 *
 *  Created on: Jun 13, 2012
 *      Author: max
 */
#ifndef DATATERMKERNELS_CU
#define DATATERMKERNELS_CU

#define SQRT2PI 2.507
#define N_GAUSS 101
#define MID_GAUSS 50
#define PI 3.1416

__device__ float GetDistance(float x1, float y1, float z1, float x2, float y2, float z2);


__global__ void K_InitGaussian (float sigma, float* gauss)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x<N_GAUSS)
	{
		gauss[x] = 1/(sigma*SQRT2PI)*expf(-0.5*(x/sigma)*(x/sigma));
	}
}

__global__ void K_InitSelectionMap (uchar* map, int width)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	map[j*width+i] = 0;
}

__global__ void K_InitColorSpace(float* pdf, int width)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	pdf[(y*N_PDF+x)*N_PDF+z] = 0;
}


__global__ void K_CalcTerms(uchar* inp_img, float* pdf, int* pos, int* neg, int width)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int x = (int)inp_img[3*(j*width+i)+0];
	int y = (int)inp_img[3*(j*width+i)+1];
	int z = (int)inp_img[3*(j*width+i)+2];

	float p = pdf[(y*N_PDF+x)*N_PDF+z];

	pos[j*width+i] = (int)(100000.0*p);
	neg[j*width+i] = (int)(100000.0*(1-p));;
}


__global__ void K_SetMap (uchar* map, int* in, int width)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	map[j*width+i] = in[j*width+i];
}


// highlight pixels in img, corresponding to 1-entries in map
__global__ void K_DrawMap (uchar* map, uchar* img, int width)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (map[j*width+i] == 1)
	{
		img[3*(j*width+i)+0] = 0;
		img[3*(j*width+i)+1] = 0;
		img[3*(j*width+i)+2] = 255;
	}
}


__global__ void K_SelectAndGauss(uchar* map, uchar* img, float* pdf, float* gauss, int x1, int y1, int rect, int width)
{

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	int ind_img;
	if (1)
	{
		ind_img = y1*width+x1;
		map[ind_img] = 1;

		uchar BGR[3];
		BGR[0] = img[3*ind_img+0];
		BGR[1] = img[3*ind_img+1];
		BGR[2] = img[3*ind_img+2];


		// calc euclidean distance to selected point
		float d = GetDistance((float)BGR[0], (float)BGR[1], (float)BGR[2], (float)x, (float)y, (float)z);
		//pdf[(y*N_PDF+x)*N_PDF+z] = d;

		__syncthreads();
		if (d<N_GAUSS)
		{
			//pdf[(y*N_PDF+x)*N_PDF+z] = gauss[(int)d];
			atomicAdd(&pdf[(y*N_PDF+x)*N_PDF+z], gauss[(int)d]);
		}
	}
}


// helper stuff

__device__ float GetDistance(float x1, float y1, float z1, float x2, float y2, float z2)
{
	return sqrtf((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2));
}


__device__ double GetElement(double* array, int i)
{
	return array[i];
}

__shared__ float d_max[N_PDF*CH_PDF];

// find maximum in array
__global__ void K_FindMax (double* pdf, double* res)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	d_max[i] = GetElement(pdf, i);
	int nTotalThreads = blockDim.x;

	while (nTotalThreads > 1)
	{
		int halfPoint = (nTotalThreads >> 1);

		if (threadIdx.x < halfPoint)
		{
			// Get the shared value stored by another thread
			double temp = d_max[threadIdx.x + halfPoint];

			if (temp > d_max[threadIdx.x])
				d_max[threadIdx.x] = temp;
		}
		__syncthreads();
		nTotalThreads = (nTotalThreads >> 1);
	}
	pdf[i] = d_max[i];

	if (i==0)
		res[0] = d_max[0];
}

__global__ void K_Normalize (double* array, double* res)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	array[i] = array[i] / res[0];
}

__global__ void K_Sum (int* array, int* res)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;
	res[j] = 0;
	__syncthreads();
	atomicAdd(&res[j], array[3*i+j]);
	//__syncthreads();
	//res[j] = sum[j];
}

// make square selection on map
__global__ void K_SelectSquare (uchar* map, int x1, int x2, int y1, int y2, int width)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i>=x1 && i<=x2 && j>=y1 && j<=y2)
		map[j*width+i] = 1;
}


#endif /* DATATERMKERNELS_CU */
