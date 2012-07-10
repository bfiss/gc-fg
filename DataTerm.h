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
	void getDataTermsMixture(int* pos, int* neg);
	void getDataTermsHSV(unsigned char* inp, int* pos);

	// size of image
	int width, height;

	// device data arrays
	unsigned char* d_selection_map;
	unsigned char* d_img;
	float* d_gaussian;
	float* d_pdf;
	int *d_pos, *d_neg;

	// kernel related
	dim3 threadsPerBlock;
	dim3 threadsPerBlock3d;
	dim3 numBlocks_3d;
	dim3 numBlocks_img;
	dim3 numBlocks_pdf;
	dim3 numBlocks_lin;
};
#endif
