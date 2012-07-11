#define N_PDF 256
#define CH_PDF 3
#include "DataTerm.h"
#include "DataTermKernels.cu"
#include <math.h>


	int* DataTerm::getPos()
	{
		return d_pos;
	}

	int* DataTerm::getNeg()
	{
		return d_neg;
	}

	// inits all datastructure needed for calculation, tell the size of the processed images
	// and the resolution of the pdf
	void DataTerm::init_data_term(int w, int h)
	{
		width = w;
		height = h;

		threadsPerBlock3d.x = 8;
		threadsPerBlock3d.y = 8;
		threadsPerBlock3d.z = 8;
		numBlocks_3d.x = N_PDF / threadsPerBlock3d.x;
		numBlocks_3d.y = N_PDF / threadsPerBlock3d.y;
		numBlocks_3d.z = N_PDF / threadsPerBlock3d.z;

		threadsPerBlock.x = 16;
		threadsPerBlock.y = 16;
		threadsPerBlock.z = 1;

		// one thread for a pixel (with 3 ch)
		numBlocks_img.x = width  / threadsPerBlock.x;
		numBlocks_img.y = height / threadsPerBlock.y;
		numBlocks_img.z = 1;


		// one thread for each pdf point per channel
		numBlocks_pdf.x = N_PDF / threadsPerBlock.x;
		numBlocks_pdf.y = 3;

		numBlocks_lin.x = 256*3 / threadsPerBlock.x;
		numBlocks_lin.y = 1;


		CUDA_SAFE_CALL(cudaMalloc((void**)&(d_img),           sizeof(unsigned char)*3*width*height));
		CUDA_SAFE_CALL(cudaMalloc((void**)&(d_gaussian),      sizeof(float)*N_PDF));
		CUDA_SAFE_CALL(cudaMalloc((void**)&(d_pdf),           sizeof(float)*N_PDF*N_PDF*N_PDF));
		CUDA_SAFE_CALL(cudaMalloc((void**)&(d_selection_map), sizeof(unsigned char)*width*height));
		CUDA_SAFE_CALL(cudaMalloc((void**)&(d_pos), sizeof(int)*width*height));
		CUDA_SAFE_CALL(cudaMalloc((void**)&(d_neg), sizeof(int)*width*height));

		reset();
	}

	// free everything
	void DataTerm::free_data_term()
	{
		CUDA_SAFE_CALL(cudaFree(d_pos));
		CUDA_SAFE_CALL(cudaFree(d_neg));
		CUDA_SAFE_CALL(cudaFree(d_img));
		CUDA_SAFE_CALL(cudaFree(d_pdf));
		CUDA_SAFE_CALL(cudaFree(d_selection_map));
		CUDA_SAFE_CALL(cudaFree(d_gaussian));
	}

	void DataTerm::setImage (unsigned char* h_img)
	{
		CUDA_SAFE_CALL(cudaMemcpy(d_img, h_img, sizeof(unsigned char)*3*width*height, cudaMemcpyHostToDevice));
	}


	void DataTerm::setSelection(int* h_select)
	{
		int* d_select;
		CUDA_SAFE_CALL(cudaMalloc((void**)&(d_select), sizeof(int)*width*height));
		CUDA_SAFE_CALL(cudaMemcpy(d_select, h_select, sizeof(int)*width*height, cudaMemcpyHostToDevice));

		K_InitSelectionMap<<<numBlocks_img, threadsPerBlock>>>(d_selection_map, width);
		K_SetMap<<<numBlocks_img, threadsPerBlock>>>(d_selection_map, d_select, width);

		CUDA_SAFE_CALL(cudaFree(d_select));
	}

	void DataTerm::reset()
	{
		K_InitSelectionMap<<<numBlocks_img, threadsPerBlock>>>(d_selection_map, width);
		K_InitColorSpace<<<numBlocks_3d, threadsPerBlock3d>>>(d_pdf, N_PDF);
	}

	// calc data terms based on selected samples
	void DataTerm::selectOnMap(int x, int y, int sigma)
	{
		int x1, y1;
		// select square around x,y with fixed sides
		for (int i=1;i<6; i++)
		{
			for (int j=0; j<6; j++)
			{
				x1 = x+i;
				y1 = y+j;

				K_InitGaussian<<<numBlocks_lin, threadsPerBlock>>>(sigma, d_gaussian);
				K_SelectAndGauss<<<numBlocks_3d, threadsPerBlock3d>>>(d_selection_map, d_img, d_pdf, d_gaussian, x1, y1, 8, width);
			}
		}

		float pdf[512];
		CUDA_SAFE_CALL(cudaMemcpy(&pdf[0], d_pdf, sizeof(float)*512, cudaMemcpyDeviceToHost));
		//int kas=0;
	}


	void DataTerm::getDataTermsMixture(int* h_pos, int* h_neg)
	{
		K_CalcTerms<<<numBlocks_img, threadsPerBlock>>>(d_img, d_pdf, d_pos, d_neg, width);
	}


	// highlight all pixels from inp that correspond to a 1-element in selection map
	void DataTerm::drawSelection(unsigned char * h_inp, unsigned char* h_out)
	{
		CUDA_SAFE_CALL(cudaMemcpy(d_img, h_inp, sizeof(unsigned char)*3*width*height, cudaMemcpyHostToDevice));
		K_DrawMap<<<numBlocks_img, threadsPerBlock>>>(d_selection_map, d_img, width);
		CUDA_SAFE_CALL(cudaMemcpy(h_out, d_img, sizeof(unsigned char)*3*width*height, cudaMemcpyDeviceToHost));
	}
