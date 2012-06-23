/*
 * GraphCut.cu
 *
 *  Created on: Jun 8, 2012
 *      Author: bruno
 */


#ifndef GRAPHCUT_CU_
#define GRAPHCUT_CU_

#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>

#include "GraphCut.h"
#include "GraphCutKernels.cu"

#define ROUND_UP(a,b) ((int)ceil((float)a/(float)b))
#define MAKE_DIVISIBLE(a,b) (b*ROUND_UP(a,b))

GlobalWrapper GC_Init(int width, int height, int * data_positive, int * data_negative, int penalty) {
	GlobalWrapper ret;
	KernelWrapper ker;

	assert(THREADS_X == 32 && THREADS_Y == 8);

	ker.g.width = width;
	ker.g.height = height;
	ker.g.size = width*height;
	ker.g.width_ex = MAKE_DIVISIBLE(width,THREADS_X);
	ker.g.height_ex = MAKE_DIVISIBLE(height,THREADS_Y);
	ker.g.size_ex = ker.g.width_ex * ker.g.height_ex;

	ker.block_x = ROUND_UP(ker.g.width_ex,THREADS_X);
	ret.block_y = ROUND_UP(ker.g.height_ex,THREADS_Y);
	ret.block_count = ROUND_UP(ker.g.size_ex,THREAD_COUNT);

	ret.penalty = penalty;

	CUDA_SAFE_CALL(cudaMalloc((void**)&(ker.g.n.edge_sink),sizeof(int)*ker.g.size_ex));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(ker.g.n.edge_u),sizeof(int)*ker.g.size_ex));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(ker.g.n.edge_d),sizeof(int)*ker.g.size_ex));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(ker.g.n.edge_l),sizeof(int)*ker.g.size_ex));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(ker.g.n.edge_r),sizeof(int)*ker.g.size_ex));
#if NEIGHBORHOOD == 8
	CUDA_SAFE_CALL(cudaMalloc((void**)&(ker.g.n.edge_ul),sizeof(int)*ker.g.size_ex));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(ker.g.n.edge_ur),sizeof(int)*ker.g.size_ex));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(ker.g.n.edge_dl),sizeof(int)*ker.g.size_ex));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(ker.g.n.edge_dr),sizeof(int)*ker.g.size_ex));
#endif
	CUDA_SAFE_CALL(cudaMalloc((void**)&(ker.g.n.height),sizeof(int)*ker.g.size_ex));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(ker.g.n.excess),sizeof(int)*ker.g.size_ex));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(ker.g.n.status),sizeof(int)*ker.g.size_ex));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(ker.active),sizeof(int)*ret.block_count));

	CUDA_SAFE_CALL(cudaMalloc((void**)&(ret.data_positive),sizeof(int)*width*height));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(ret.data_negative),sizeof(int)*width*height));
	CUDA_SAFE_CALL(cudaMemcpy(ret.data_positive,data_positive,sizeof(int)*width*height,cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ret.data_negative,data_negative,sizeof(int)*width*height,cudaMemcpyHostToDevice));

	dim3 block(THREADS_X,THREADS_Y,1);
	dim3 grid(ker.block_x,ret.block_y,1);

	InitGraph<<<grid,block>>>(ker,ret.data_positive,ret.data_negative,penalty);
	cutilCheckMsg("InitGraph kernel launch failure");

	ret.k = ker;

	return ret;
}

void GC_Update(GlobalWrapper gw, int * data) {
	assert(gw.k.block_x);

}

#define ACTIVITY_CHECK_FREQUENCY 10
#define PUSHES_PER_RELABEL 1

void GC_Optimize(GlobalWrapper gw, int * label) {
	dim3 block(THREADS_X,THREADS_Y,1);
	dim3 grid(gw.k.block_x,gw.block_y,1);
	
	int * zero_arr = (int *) malloc(8*sizeof(int));
	zero_arr[0] = 0;
	int * h_alive = (int *) malloc(8*sizeof(int));
	int * d_alive;
	CUDA_SAFE_CALL(cudaMalloc((void**)&(d_alive),8*sizeof(int)));
	
	int counter = 0;
	
	const char * error;
	
	int * d_heights;
	CUDA_SAFE_CALL(cudaMalloc((void**)&(d_heights),(gw.k.g.size_ex+10)*sizeof(int)));

	//bool dbg_verify_no_more_pushes = false;

	
	unsigned int timer = 0;
	CUT_SAFE_CALL(cutCreateTimer(&timer));
	CUT_SAFE_CALL(cutStartTimer(timer));

	while(1) {
		int skip = counter % ACTIVITY_CHECK_FREQUENCY;
		
		++counter;
		
		if(!skip)
			CUDA_SAFE_CALL(cudaMemcpy(d_alive,zero_arr,8*sizeof(int),cudaMemcpyHostToDevice));
		
		Relabel<<<grid,block>>>(gw.k, skip);
		/*cutilCheckMsg("Relabel kernel launch failure");error = cudaGetErrorString(cudaPeekAtLastError());
		printf("%s\nwith %d iters.\n", error,counter);
		error = cudaGetErrorString(cudaThreadSynchronize());
		printf("%s\n", error);*/
		
		//CUDA_SAFE_CALL(cudaThreadSynchronize());
		
		Push<<<grid,block>>>(gw.k, PUSHES_PER_RELABEL, skip, d_alive);
		/*cutilCheckMsg("Push kernel launch failure");error = cudaGetErrorString(cudaPeekAtLastError());
		printf("%s\nwith %d iters.\n", error,counter);
		error = cudaGetErrorString(cudaThreadSynchronize());
		printf("%s\n", error);*/
		
		//CUDA_SAFE_CALL(cudaThreadSynchronize());
		
		if(!skip)
			CUDA_SAFE_CALL(cudaMemcpy(h_alive,d_alive,8*sizeof(int),cudaMemcpyDeviceToHost));
	
	
		if(!skip) {
			UpdateActivity<<<grid,block>>>(gw.k.g.n.status, gw.k.active, gw.k.block_x, gw.k.g.width_ex);
			/*cutilCheckMsg("UpdateActivity kernel launch failure");error = cudaGetErrorString(cudaPeekAtLastError());
			printf("%s\nwith %d iters.\n", error,counter);
			error = cudaGetErrorString(cudaThreadSynchronize());
			printf("%s\n", error);*/
		}
		//CUDA_SAFE_CALL(cudaThreadSynchronize());
		
		if(!h_alive[0]) {
			//dbg_verify_no_more_pushes = true;
			break;
		} /*else if ( dbg_verify_no_more_pushes ) {
			printf("Became alive after being dead!!\n");
			assert(false);
		}*/
		if(counter > 3000) {
			printf("Too long inside the main loop\n");
			break;
		}
	}
	
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL(cutStopTimer(timer));
	printf("Graph Cut used %d iterations and %f milliseconds.\n", counter, cutGetTimerValue(timer));
	CUT_SAFE_CALL(cutDeleteTimer(timer));
	
	CUDA_SAFE_CALL(cudaFree(d_heights));

	int * d_label;
	CUDA_SAFE_CALL(cudaMalloc((void**)&(d_label),sizeof(int)*gw.k.g.width*gw.k.g.height));
	CUDA_SAFE_CALL(cudaMemcpy(d_label,label,sizeof(int)*gw.k.g.width*gw.k.g.height,cudaMemcpyHostToDevice));
	InitLabels<<<grid,block>>>(gw.k, d_label);
	cutilCheckMsg("InitLabels kernel launch failure");
	
	while(1){
		CUDA_SAFE_CALL(cudaMemcpy(d_alive,zero_arr,8*sizeof(int),cudaMemcpyHostToDevice));
		SpreadLabels<<<grid,block>>>(gw.k, d_label, d_alive);
		cutilCheckMsg("SpreadLabels kernel launch failure");
		CUDA_SAFE_CALL(cudaMemcpy(h_alive,d_alive,8*sizeof(int),cudaMemcpyDeviceToHost));
		if(!h_alive[0])
			break;
	}
	CUDA_SAFE_CALL(cudaMemcpy(label,d_label,sizeof(int)*gw.k.g.width*gw.k.g.height,cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_label));
	CUDA_SAFE_CALL(cudaFree(d_alive));
	free(zero_arr);
	free(h_alive);
}

void GC_End(GlobalWrapper * gw) {
	GlobalWrapper clean;

	CUDA_SAFE_CALL(cudaFree(gw->k.g.n.edge_sink));
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
	CUDA_SAFE_CALL(cudaFree(gw->k.active));
	
	CUDA_SAFE_CALL(cudaFree(gw->data_positive));
	CUDA_SAFE_CALL(cudaFree(gw->data_negative));

	clean.k.block_x = clean.block_y = clean.k.g.width = clean.k.g.height = 0;
	*gw = clean;
}


#endif /* GRAPHCUT_CU_ */
