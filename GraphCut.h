/*
 * GraphCut.h
 *
 *  Created on: Jun 8, 2012
 *      Author: bruno
 */

#ifndef GRAPHCUT_H_
#define GRAPHCUT_H_

#include <cuda.h>
#include <cutil.h>
#include <cutil_inline.h>

#define NEIGHBORHOOD 4

//#define DEBUG_MODE

#define SPREAD_ZEROS

#define THREADS_X 32
#define THREADS_Y 8
#define THREAD_COUNT ((THREADS_X)*(THREADS_Y))

struct NodeWrapper {
	int * edge_l;
	int * edge_r;
	int * edge_u;
	int * edge_d;
#if NEIGHBORHOOD == 8
	int * edge_ul;
	int * edge_dr;
	int * edge_ur;
	int * edge_dl;
#endif
	int * height;
	int * excess;
	int * status;
	int * comp_h;
	int * comp_n;
};

struct GraphWrapper {
	int width;
	int height;
	int size;
	int width_ex;
	int height_ex;
	int size_ex;
	NodeWrapper n;
};

struct KernelWrapper {
	int block_x;
	int * active;
	GraphWrapper g;
};

struct GlobalWrapper {
	int block_y;
	int block_count;
	int * data_positive;
	int * data_negative;
	int penalty;
	KernelWrapper k;
};

/*
KernelWrapper GC_Init(int width, int height, int * data);
void GC_Update(int * data, KernelWrapper k);
void GC_Optimize(KernelWrapper k, int * label);
void GC_End(KernelWrapper * k);
*/

#endif /* GRAPHCUT_H_ */
