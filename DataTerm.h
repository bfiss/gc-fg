/*! @file
 * DataTerm.h
 * Definitions and data structures for data term estimation algorithms.
 */
#ifndef DATATERM_H
#define DATATERM_H
namespace DataTerm
{
	int* getPos();
	int* getNeg();
	void init_data_term(int w, int h);
	void free_data_term();
	void setImage (unsigned char* h_img);
	void setSelection(int* h_select);
	void reset();
	void selectOnMap(int x, int y, int sigma);
	void drawSelection(unsigned char* inp, unsigned char* out);
	void getDataTermsMixture();
	void setSigmaGaussian(int sigma);

	// size of image
	int width;		//!<width of image
	int height;		//!<height of image

	// device data arrays
	unsigned char* d_selection_map;	//!<device pointer for image selection map
	unsigned char* d_img;			//!<device pointer for image data used for calculations
	float* d_gaussian;				//!<device pointer for gaussian envelope
	float* d_pdf;					//!<device pointer for estimated probability density of class membership based on color information
	int *d_pos; 					//!<device pointer for positive data terms
	int *d_neg; 					//!<device pointer for negative data terms

	// kernel related
	dim3 threadsPerBlock;			//!< for 2d processing
	dim3 threadsPerBlock3d;			//!< threads per block for pdf update (256*256*256 data points)
	dim3 numBlocks_3d;				//!< numblocks for pdf update (256*256*256 data points)
	dim3 numBlocks_img;				//!< numblocks for image processing (width*height threads)
	dim3 numBlocks_lin;				//!< numblocks for linear array processing
};
#endif
