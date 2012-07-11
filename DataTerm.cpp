/*! @file
 * Functions for Data Term calculation and Image highlighting.
 */

#define N_PDF 256 		//!>Size of colorspace used for class membership probability density. The used colorspace is 256*256*256
#define DEFAULT_SIGMA 5	//!>default std deviation value for gaussian envelope

#include "DataTerm.h"
#include "DataTermKernels.cu"


	int* DataTerm::getPos()
	{
		/*!
		 * Returns device pointer for positive data points. Used inside graph cuts initialization.
		 */
		return d_pos;
	}

	int* DataTerm::getNeg()
	{
		/*!
		 * Returns device pointer for negative data points. Used inside graph cuts initialization.
		 */
		return d_neg;
	}

	void DataTerm::init_data_term(int w, int h)
	{
		/*!
		 * Inits all datastructure needed for calculation, tell the size of the processed image and the resolution of the pdf.
		 */
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

		numBlocks_lin.x = 256*3 / threadsPerBlock.x;
		numBlocks_lin.y = 1;


		CUDA_SAFE_CALL(cudaMalloc((void**)&(d_img),           sizeof(uchar)*3*width*height));
		CUDA_SAFE_CALL(cudaMalloc((void**)&(d_gaussian),      sizeof(float)*N_PDF));
		CUDA_SAFE_CALL(cudaMalloc((void**)&(d_pdf),           sizeof(float)*N_PDF*N_PDF*N_PDF));
		CUDA_SAFE_CALL(cudaMalloc((void**)&(d_selection_map), sizeof(uchar)*width*height));
		// will be freed in graph cuts
		CUDA_SAFE_CALL(cudaMalloc((void**)&(d_pos), sizeof(int)*width*height));
		CUDA_SAFE_CALL(cudaMalloc((void**)&(d_neg), sizeof(int)*width*height));

		K_InitGaussian<<<numBlocks_lin, threadsPerBlock>>>(DEFAULT_SIGMA, d_gaussian);
		reset();
	}

	void DataTerm::free_data_term()
	{
		/*!
		 * free device memory
		 */
		//CUDA_SAFE_CALL(cudaFree(d_pos));
		//CUDA_SAFE_CALL(cudaFree(d_neg));
		CUDA_SAFE_CALL(cudaFree(d_img));
		CUDA_SAFE_CALL(cudaFree(d_pdf));
		CUDA_SAFE_CALL(cudaFree(d_selection_map));
		CUDA_SAFE_CALL(cudaFree(d_gaussian));
	}

	void DataTerm::setImage (unsigned char* h_img)
	{
		/*!
		 * Load image data to device.
		 */
		CUDA_SAFE_CALL(cudaMemcpy(d_img, h_img, sizeof(uchar)*3*width*height, cudaMemcpyHostToDevice));
	}

	void DataTerm::reset()
	{
		/*!
		 * Set selection map and pdf to 0.
		 */
		K_InitSelectionMap<<<numBlocks_img, threadsPerBlock>>>(d_selection_map, width);
		K_InitColorSpace<<<numBlocks_3d, threadsPerBlock3d>>>(d_pdf, N_PDF);
	}

	void DataTerm::setSigmaGaussian(int sigma)
	{
		/*!
		 * Calculate gaussian array with given variance.
		 */
		K_InitGaussian<<<numBlocks_lin, threadsPerBlock>>>(sigma, d_gaussian);
	}

	void DataTerm::selectOnMap(int x, int y, int sigma)
	{
		/*!
		 * Select data points from image. For each point add a gaussian kernel, with variance set in setSigmaGaussian, to the pdf.
		 * Each datapoint is processed iterative. The update of the density estimation is run parallel.
		 */
		int x1, y1;
		// select square around x,y with fixed sides
		int size = 5;
		for (int i=0;i<size; i++)
		{
			for (int j=0; j<size; j++)
			{
				x1 = x+i;
				y1 = y+j;

				K_SelectAndGauss<<<numBlocks_3d, threadsPerBlock3d>>>(d_selection_map, d_img, d_pdf, d_gaussian, x1, y1, 8, width);
			}
		}

		float pdf[512];
		CUDA_SAFE_CALL(cudaMemcpy(&pdf[0], d_pdf, sizeof(float)*512, cudaMemcpyDeviceToHost));
	}

	void DataTerm::getDataTermsMixture()
	{
		/*!
		 * Use the calculated pdf to calculate the data terms for a given image (set by setImage). To a pixel (x,y) corresponds the
		 * color vector c=(b(x,y), g(x,y), r(x,y)). The probability for the pixel belonging to the selected class is then pdf(c).
		 * Positive data term is p, negative is 1-p. Both are stored in d_pos and d_neg.
		 */
		K_CalcTerms<<<numBlocks_img, threadsPerBlock>>>(d_img, d_pdf, d_pos, d_neg, width);
	}

	void DataTerm::setSelection(int* h_select)
		{
			/*!
			 * Set selection map on device.
			 * @param[in] h_select Selection / labeling matrix.
			 */
			int* d_select;
			CUDA_SAFE_CALL(cudaMalloc((void**)&(d_select), sizeof(int)*width*height));
			CUDA_SAFE_CALL(cudaMemcpy(d_select, h_select, sizeof(int)*width*height, cudaMemcpyHostToDevice));

			K_InitSelectionMap<<<numBlocks_img, threadsPerBlock>>>(d_selection_map, width);
			K_SetMap<<<numBlocks_img, threadsPerBlock>>>(d_selection_map, d_select, width);

			CUDA_SAFE_CALL(cudaFree(d_select));
		}

	void DataTerm::drawSelection(uchar* h_inp, uchar* h_out)
	{
		/*
		 * Highlight all pixels in input image data that correspond to a 1-entry in selection map.
		 * @param[in] h_inp Input image rawdata
		 * @param[out] h_out processed image data
		 */
		CUDA_SAFE_CALL(cudaMemcpy(d_img, h_inp, sizeof(uchar)*3*width*height, cudaMemcpyHostToDevice));
		K_DrawMap<<<numBlocks_img, threadsPerBlock>>>(d_selection_map, d_img, width);
		CUDA_SAFE_CALL(cudaMemcpy(h_out, d_img, sizeof(uchar)*3*width*height, cudaMemcpyDeviceToHost));
	}
