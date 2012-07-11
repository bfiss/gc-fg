
/*!
 * \file GraphCut.cu
 *
 * \author bruno
 * \date Jun 8, 2012
 *
 * CUDA source file that contains the interface of Graph Cut, and is responsible for host sided computations.
 */

#ifndef GRAPHCUT_CU_
#define GRAPHCUT_CU_

#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>

#include "GraphCut.h"
#include "GraphCutKernels.cu"

//! Sets the data terms
/*!
 * This function expects \a data_positive and \a data_negative to be
 * pointers to device allocated memory holding the data terms. It updates
 * the current data terms for the Graph Cut algorithm, without optimizing. This
 * functions requires the graph to be already initialized (via a call to GC_Init).
 */
void GC_SetDataterms(GlobalWrapper* gw, int* data_positive, int* data_negative);

//! Sets the edge values between neighbors
/*!
 * This function expects its parameters to be
 * pointers to device allocated memory holding the edge values. It updates
 * the current edge values for the Graph Cut algorithm, without optimizing. This
 * functions requires the graph to be already initialized (via a call to GC_Init).
 *
 */
void GC_SetEdges(GlobalWrapper* gw, int *, int *, int *, int *
#if NEIGHBORHOOD == 8
		, int *, int *, int *, int *
#endif
		);

//! Sets up the internal graph to allow graph cut to be run.
/*!
 * This function should be called after having set data terms (and possibly edge values), either within
 * GC_Init, or with the specific functions. It prepares the graph structure used in the algorithm, allowing
 * optimizations to be run afterwards.
 *
 */
void GC_SetGraph(GlobalWrapper gw);

#ifdef DEBUG_MODE

NodeWrapper host_copy;

static void initialize_graph(GraphWrapper gw) {
	host_copy.height = (int *) malloc(sizeof(int)*gw.width_ex*gw.height_ex);
	host_copy.excess = (int *) malloc(sizeof(int)*gw.width_ex*gw.height_ex);
	host_copy.status = (int *) malloc(sizeof(int)*gw.width_ex*gw.height_ex);
	host_copy.comp_h = (int *) malloc(sizeof(int)*gw.width_ex*gw.height_ex);
	host_copy.comp_n = (int *) malloc(sizeof(int)*gw.width_ex*gw.height_ex);
	host_copy.edge_l = (int *) malloc(sizeof(int)*gw.width_ex*gw.height_ex);
	host_copy.edge_r = (int *) malloc(sizeof(int)*gw.width_ex*gw.height_ex);
	host_copy.edge_u = (int *) malloc(sizeof(int)*gw.width_ex*gw.height_ex);
	host_copy.edge_d = (int *) malloc(sizeof(int)*gw.width_ex*gw.height_ex);
#if NEIGHBORHOOD == 8
	host_copy.edge_ul = (int *) malloc(sizeof(int)*gw.width_ex*gw.height_ex);
	host_copy.edge_dr = (int *) malloc(sizeof(int)*gw.width_ex*gw.height_ex);
	host_copy.edge_ur = (int *) malloc(sizeof(int)*gw.width_ex*gw.height_ex);
	host_copy.edge_dl = (int *) malloc(sizeof(int)*gw.width_ex*gw.height_ex);
#endif
}

static void update_graph(GraphWrapper gw) {
	CUDA_SAFE_CALL(cudaMemcpy(host_copy.height,gw.n.height,sizeof(int)*gw.width_ex*gw.height_ex,cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(host_copy.excess,gw.n.excess,sizeof(int)*gw.width_ex*gw.height_ex,cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(host_copy.status,gw.n.status,sizeof(int)*gw.width_ex*gw.height_ex,cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(host_copy.comp_h,gw.n.comp_h,sizeof(int)*gw.width_ex*gw.height_ex,cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(host_copy.comp_n,gw.n.comp_n,sizeof(int)*gw.width_ex*gw.height_ex,cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(host_copy.edge_l,gw.n.edge_l,sizeof(int)*gw.width_ex*gw.height_ex,cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(host_copy.edge_r,gw.n.edge_r,sizeof(int)*gw.width_ex*gw.height_ex,cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(host_copy.edge_u,gw.n.edge_u,sizeof(int)*gw.width_ex*gw.height_ex,cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(host_copy.edge_d,gw.n.edge_d,sizeof(int)*gw.width_ex*gw.height_ex,cudaMemcpyDeviceToHost));
#if NEIGHBORHOOD == 8
	CUDA_SAFE_CALL(cudaMemcpy(host_copy.edge_ul,gw.n.edge_ul,sizeof(int)*gw.width_ex*gw.height_ex,cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(host_copy.edge_dr,gw.n.edge_dr,sizeof(int)*gw.width_ex*gw.height_ex,cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(host_copy.edge_ur,gw.n.edge_ur,sizeof(int)*gw.width_ex*gw.height_ex,cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(host_copy.edge_dl,gw.n.edge_dl,sizeof(int)*gw.width_ex*gw.height_ex,cudaMemcpyDeviceToHost));
#endif
}

static void print_graph(GraphWrapper gw) {
	update_graph(gw);
	for(int i = 0; i < gw.height; ++i) {
		for(int j = 0; j < gw.width; ++j) {
			printf("%2d ",host_copy.height[i*gw.width_ex + j]);
		}
		printf(" | ");
		for(int j = 0; j < gw.width; ++j) {
			printf("%2d ",host_copy.excess[i*gw.width_ex + j]);
		}
		printf(" | ");
		for(int j = 0; j < gw.width; ++j) {
			printf("%2d ",host_copy.edge_u[i*gw.width_ex + j]);
		}
		printf(" | ");
		for(int j = 0; j < gw.width; ++j) {
			printf("%2d ",host_copy.edge_l[i*gw.width_ex + j]);
		}
		printf(" | ");
		for(int j = 0; j < gw.width; ++j) {
			printf("%2d ",host_copy.edge_d[i*gw.width_ex + j]);
		}
		printf(" | ");
		for(int j = 0; j < gw.width; ++j) {
			printf("%2d ",host_copy.edge_r[i*gw.width_ex + j]);
		}
		printf(" | ");
		for(int j = 0; j < gw.width; ++j) {
			printf("%2d ",host_copy.comp_h[i*gw.width_ex + j]);
		}
		printf(" | ");
		for(int j = 0; j < gw.width; ++j) {
			printf("%2d ",host_copy.comp_n[i*gw.width_ex + j]);
		}
		printf("\n");
	}
}

static void free_graph(GraphWrapper gw) {
	free(host_copy.height);
	free(host_copy.excess);
	free(host_copy.status);
	free(host_copy.comp_h);
	free(host_copy.comp_n);
	free(host_copy.edge_l);
	free(host_copy.edge_r);
	free(host_copy.edge_u);
	free(host_copy.edge_d);
#if NEIGHBORHOOD == 8
	free(host_copy.edge_ul);
	free(host_copy.edge_dr);
	free(host_copy.edge_ur);
	free(host_copy.edge_dl);
#endif
}

#endif

/*! \def ROUND_UP(a,b)
 * \brief Rounds \a a divided by \a b up.
 */
#define ROUND_UP(a,b) ((int)ceil((float)a/(float)b))

/*! \def MAKE_DIVISIBLE(a,b)
 * \brief Returns the first multiple of \a b that is bigger or equal to \a a
 */
#define MAKE_DIVISIBLE(a,b) (b*ROUND_UP(a,b))

//! Initializes the internal structure of Graph Cut
/*!
 * This function should be the first called in order to perform a Graph Cut.
 * It must receive the \a width and \a height of the picture, network or field to be cut.
 * It may also receive the data terms, which are used as the capacities between each node and the source and
 * sink nodes, respectively, and a \a penalty, which will be the capacity of edges between neighbor nodes.
 * The last possibility is to explicitly give the capacities of edges between neighbors, using one array per direction.
 * In this case, \a penalty is ignored. Data terms and edge values are expected to be in
 * device memory, and to have a size equal to the picture/field size.
 *
 */
GlobalWrapper GC_Init(int width, int height, int * data_positive = NULL,
		int * data_negative = NULL, int penalty = 0, int * up = NULL,
		int * down = NULL, int * left = NULL, int * right = NULL
#if NEIGHBORHOOD == 8
		, int * upleft = NULL, int * upright = NULL, int * downleft = NULL,
		int * downright = NULL
#endif
		) {
	GlobalWrapper ret;
	KernelWrapper ker;

	assert(THREADS_X == 32 && THREADS_Y == 8);

	ker.g.width = width;
	ker.g.height = height;
	ker.g.size = width * height;
	ker.g.width_ex = MAKE_DIVISIBLE(width,THREADS_X);
	ker.g.height_ex = MAKE_DIVISIBLE(height,THREADS_Y);
	ker.g.size_ex = ker.g.width_ex * ker.g.height_ex;

	ker.block_x = ROUND_UP(ker.g.width_ex,THREADS_X);
	ret.block_y = ROUND_UP(ker.g.height_ex,THREADS_Y);
	ret.block_count = ROUND_UP(ker.g.size_ex,THREAD_COUNT);

	ret.penalty = penalty;

	ret.varying_edges = up != NULL;

	CUDA_SAFE_CALL(
			cudaMalloc((void**)&(ker.g.n.edge_u),sizeof(int)*ker.g.size_ex));
	CUDA_SAFE_CALL(
			cudaMalloc((void**)&(ker.g.n.edge_d),sizeof(int)*ker.g.size_ex));
	CUDA_SAFE_CALL(
			cudaMalloc((void**)&(ker.g.n.edge_l),sizeof(int)*ker.g.size_ex));
	CUDA_SAFE_CALL(
			cudaMalloc((void**)&(ker.g.n.edge_r),sizeof(int)*ker.g.size_ex));
#if NEIGHBORHOOD == 8
	CUDA_SAFE_CALL(
			cudaMalloc((void**)&(ker.g.n.edge_ul),sizeof(int)*ker.g.size_ex));
	CUDA_SAFE_CALL(
			cudaMalloc((void**)&(ker.g.n.edge_ur),sizeof(int)*ker.g.size_ex));
	CUDA_SAFE_CALL(
			cudaMalloc((void**)&(ker.g.n.edge_dl),sizeof(int)*ker.g.size_ex));
	CUDA_SAFE_CALL(
			cudaMalloc((void**)&(ker.g.n.edge_dr),sizeof(int)*ker.g.size_ex));
#endif
	CUDA_SAFE_CALL(
			cudaMalloc((void**)&(ker.g.n.height),sizeof(int)*ker.g.size_ex));
	CUDA_SAFE_CALL(
			cudaMalloc((void**)&(ker.g.n.excess),sizeof(int)*ker.g.size_ex));
	CUDA_SAFE_CALL(
			cudaMalloc((void**)&(ker.g.n.status),sizeof(int)*ker.g.size_ex));
	CUDA_SAFE_CALL(
			cudaMalloc((void**)&(ker.g.n.comp_h),sizeof(int)*ker.g.size_ex));
	CUDA_SAFE_CALL(
			cudaMalloc((void**)&(ker.g.n.comp_n),sizeof(int)*ker.g.size_ex));
	CUDA_SAFE_CALL(
			cudaMalloc((void**)&(ker.active),sizeof(int)*ret.block_count));

	ret.k = ker;

	if (data_positive != NULL) {

		GC_SetDataterms(&ret, data_positive, data_negative);
		if (ret.varying_edges)
			GC_SetEdges(&ret, up, down, left, right
#if NEIGHBORHOOD == 8
					, upleft, upright, downleft, downright
#endif
					);
		GC_SetGraph(ret);
	}

	return ret;
}

void GC_SetDataterms(GlobalWrapper* gw, int* data_positive,
		int* data_negative) {
	gw->data_positive = data_positive;
	gw->data_negative = data_negative;
}

void GC_SetEdges(GlobalWrapper* gw, int * up = NULL, int * down = NULL,
		int * left = NULL, int * right = NULL
#if NEIGHBORHOOD == 8
		, int * upleft = NULL, int * upright = NULL, int * downleft = NULL,
		int * downright = NULL
#endif
		) {
	gw->varying_edges = true;
	gw->up = up;
	gw->down = down;
	gw->left = left;
	gw->right = right;
#if NEIGHBORHOOD == 8
	gw->upleft = upleft;
	gw->upright = upright;
	gw->downleft = downleft;
	gw->downright = downright;
#endif
}

//! This function sets every edge capacity to \a p.
/*!
 *
 *
 */
void GC_SetPenalty(GlobalWrapper* gw, int p) {
	gw->varying_edges = false;
	gw->penalty = p;
}

void GC_SetGraph(GlobalWrapper gw) {

	dim3 block(THREADS_X, THREADS_Y, 1);
	dim3 grid(gw.k.block_x, gw.block_y, 1);

	printf("pen %d\n", gw.penalty);
	if (gw.varying_edges) {
		InitGraphVarEdges<<<grid,block>>>(gw.k, gw.data_positive, gw.data_negative, gw.up,
				gw.down, gw.left, gw.right
#if NEIGHBORHOOD == 8
				, gw.upleft, gw.upright, gw.downleft, gw.downright
#endif
				);
		cutilCheckMsg("InitGraphVarEdges kernel launch failure");
	} else {
		InitGraph<<<grid,block>>>(gw.k, gw.data_positive, gw.data_negative, gw.penalty);
		cutilCheckMsg("InitGraph kernel launch failure");
	}
}

/*
 void GC_Update(GlobalWrapper gw, int * data) {
 assert(gw.k.block_x);

 }*/

/*! \def ACTIVITY_CHECK_FREQUENCY
 * \brief How often inactive blocks are used in the kernels.
 *
 * This parameter controls when blocks that were marked as inactive are
 * used again in Push and Relabel. After this many iterations, all blocks are used again.
 */
#define ACTIVITY_CHECK_FREQUENCY 10

/*! \def GLOBAL_RELABEL_FREQUENCY
 * \brief How often Global Relabel is run.
 *
 * This parameter controls after how many iterations of Push and Relabel the Global Relabel is executed.
 */
#define GLOBAL_RELABEL_FREQUENCY 20 // 150
/*! \def FIRST_GLOBAL_RELABEL
 * \brief At which iteration Global Relabel is first run.
 *
 * This parameter controls at which iteration Global Relabel is first run.
 */
#define FIRST_GLOBAL_RELABEL 15

/*! \def PUSHES_PER_RELABEL
 * \brief Controls how many pushes are executed per relabel.
 *
 * This parameter controls the number of pushes executed consecutively inside the same Push kernel.
 * It defines how many pushes happen for every relabel.
 */
#define PUSHES_PER_RELABEL 4

//! Runs Graph Cut on given data
/*!
 * This function expects a Global Wrapper \a gw properly initialized
 * using GC_Init and the other set functions. It then performs a Graph Cut on
 * the data given.
 *
 */
void GC_Optimize(GlobalWrapper gw, int * label) {
	dim3 block(THREADS_X, THREADS_Y, 1);
	dim3 grid(gw.k.block_x, gw.block_y, 1);

#ifdef DEBUG_MODE
	initialize_graph(gw.k.g);
#endif

	int * zero_arr = (int *) malloc(8 * sizeof(int));
	zero_arr[0] = 0;
	int * h_alive = (int *) malloc(8 * sizeof(int));
	int * d_alive;
	CUDA_SAFE_CALL(cudaMalloc((void**)&(d_alive),8*sizeof(int)));

	int counter = 0;

	//const char * error;

	int * d_heights;
	CUDA_SAFE_CALL(
			cudaMalloc((void**)&(d_heights),(gw.k.g.size_ex+10)*sizeof(int)));

	//bool dbg_verify_no_more_pushes = false;

#ifdef DEBUG_MODE
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	printf("Starting out:\n");
	print_graph(gw.k.g);
#endif

	unsigned int timer = 0;
	CUT_SAFE_CALL(cutCreateTimer(&timer));
	CUT_SAFE_CALL(cutStartTimer(timer));

	while (1) {
		int skip = counter % ACTIVITY_CHECK_FREQUENCY;

		++counter;

		if (!skip)
			CUDA_SAFE_CALL(
					cudaMemcpy(d_alive,zero_arr,8*sizeof(int),cudaMemcpyHostToDevice));

		Relabel<<<grid,block>>>(gw.k, skip);
		/*cutilCheckMsg("Relabel kernel launch failure");error = cudaGetErrorString(cudaPeekAtLastError());
		 printf("%s\nwith %d iters.\n", error,counter);
		 error = cudaGetErrorString(cudaThreadSynchronize());
		 printf("%s\n", error);*/

		//CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifdef DEBUG_MODE
		CUDA_SAFE_CALL(cudaThreadSynchronize());
		printf("After Relabel:\n");
		print_graph(gw.k.g);
#endif

		Push<<<grid,block>>>(gw.k, PUSHES_PER_RELABEL, skip, d_alive);
		/*cutilCheckMsg("Push kernel launch failure");error = cudaGetErrorString(cudaPeekAtLastError());
		 printf("%s\nwith %d iters.\n", error,counter);
		 error = cudaGetErrorString(cudaThreadSynchronize());
		 printf("%s\n", error);*/

		//CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifdef DEBUG_MODE
		CUDA_SAFE_CALL(cudaThreadSynchronize());
		printf("After Push:\n");
		print_graph(gw.k.g);
#endif

		if (!skip)
			CUDA_SAFE_CALL(
					cudaMemcpy(h_alive,d_alive,8*sizeof(int),cudaMemcpyDeviceToHost));

		if (!skip) {
UpdateActivity		<<<grid,block>>>(gw.k.g.n.status, gw.k.active, gw.k.block_x, gw.k.g.width_ex);
		/*cutilCheckMsg("UpdateActivity kernel launch failure");error = cudaGetErrorString(cudaPeekAtLastError());
		 printf("%s\nwith %d iters.\n", error,counter);
		 error = cudaGetErrorString(cudaThreadSynchronize());
		 printf("%s\n", error);*/
	}
	//CUDA_SAFE_CALL(cudaThreadSynchronize());

		if (!h_alive[0]) {
			//dbg_verify_no_more_pushes = true;
			break;
		}
		if (!((counter - FIRST_GLOBAL_RELABEL) % GLOBAL_RELABEL_FREQUENCY)) {
InitGlobalRelabel<<<grid,block>>>(gw.k);
						cutilCheckMsg("InitGlobalRelabel kernel launch failure");
			int iter_gr = 0;
			while (1) {
				//if(!(iter_gr % GLOBAL_RELABEL_CHECK_FREQUENCY))
				CUDA_SAFE_CALL(
						cudaMemcpy(d_alive,zero_arr,8*sizeof(int),cudaMemcpyHostToDevice));
GlobalRelabel<<<grid,block>>>(gw.k, d_alive);
								cutilCheckMsg("GlobalRelabel kernel launch failure");
				//if(!(iter_gr % GLOBAL_RELABEL_CHECK_FREQUENCY))
				CUDA_SAFE_CALL(
						cudaMemcpy(h_alive,d_alive,8*sizeof(int),cudaMemcpyDeviceToHost));
				if (!h_alive[0])
					break;
				iter_gr++;
			}
			printf("%d iterations inside global relabel\n", iter_gr);
			h_alive[0] = 1;
#ifdef DEBUG_MODE
			CUDA_SAFE_CALL(cudaThreadSynchronize());
			printf("After Global Relabel:\n");
			print_graph(gw.k.g);
#endif
		}

		if (!(counter % 500))
			printf("counter: %d\n", counter);
	}

#ifdef DEBUG_MODE
	free_graph(gw.k.g);
#endif

	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL(cutStopTimer(timer));
	printf("Graph Cut used %d iterations and %f milliseconds.\n", counter,
			cutGetTimerValue(timer));
	CUT_SAFE_CALL(cutDeleteTimer(timer));

	CUDA_SAFE_CALL(cudaFree(d_heights));

	int * d_label;
	CUDA_SAFE_CALL(
			cudaMalloc((void**)&(d_label),sizeof(int)*gw.k.g.width*gw.k.g.height));
	CUDA_SAFE_CALL(
			cudaMemcpy(d_label,label,sizeof(int)*gw.k.g.width*gw.k.g.height,cudaMemcpyHostToDevice));
	InitLabels<<<grid,block>>>(gw.k, d_label);
	cutilCheckMsg("InitLabels kernel launch failure");

	int spreadLabelsCounter = 0;
	while (1) {
		spreadLabelsCounter++;
		CUDA_SAFE_CALL(
				cudaMemcpy(d_alive,zero_arr,8*sizeof(int),cudaMemcpyHostToDevice));
		SpreadLabels<<<grid,block>>>(gw.k, d_label, d_alive);
				cutilCheckMsg("SpreadLabels kernel launch failure");
		CUDA_SAFE_CALL(
				cudaMemcpy(h_alive,d_alive,8*sizeof(int),cudaMemcpyDeviceToHost));
		if (!h_alive[0])
			break;
	}
	printf("%d iterations inside Spreadlabels\n", spreadLabelsCounter);
	CUDA_SAFE_CALL(
			cudaMemcpy(label,d_label,sizeof(int)*gw.k.g.width*gw.k.g.height,cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_label));
	CUDA_SAFE_CALL(cudaFree(d_alive));
	free(h_alive);
	free(zero_arr);
}

//! Terminates Graph Cut and frees allocated memory
/*!
 * This function should be called in order to properly terminate Graph Cut
 * and free the structures allocated by it.
 *
 */
void GC_End(GlobalWrapper * gw) {
	GlobalWrapper clean;

	CUDA_SAFE_CALL(cudaFree(gw->k.g.n.edge_u));
	CUDA_SAFE_CALL(cudaFree(gw->k.g.n.edge_d));
	CUDA_SAFE_CALL(cudaFree(gw->k.g.n.edge_l));
	CUDA_SAFE_CALL(cudaFree(gw->k.g.n.edge_r));
#if NEIGHBORHOOD == 8
	CUDA_SAFE_CALL(cudaFree(gw->k.g.n.edge_ul));
	CUDA_SAFE_CALL(cudaFree(gw->k.g.n.edge_ur));
	CUDA_SAFE_CALL(cudaFree(gw->k.g.n.edge_dl));
	CUDA_SAFE_CALL(cudaFree(gw->k.g.n.edge_dr));
#endif
	CUDA_SAFE_CALL(cudaFree(gw->k.g.n.height));
	CUDA_SAFE_CALL(cudaFree(gw->k.g.n.excess));
	CUDA_SAFE_CALL(cudaFree(gw->k.g.n.status));
	CUDA_SAFE_CALL(cudaFree(gw->k.g.n.comp_h));
	CUDA_SAFE_CALL(cudaFree(gw->k.g.n.comp_n));
	CUDA_SAFE_CALL(cudaFree(gw->k.active));

	/*CUDA_SAFE_CALL(cudaFree(gw->data_positive));
	 CUDA_SAFE_CALL(cudaFree(gw->data_negative));*/

	clean.k.block_x = clean.block_y = clean.k.g.width = clean.k.g.height = 0;
	*gw = clean;
}

#endif /* GRAPHCUT_CU_ */
