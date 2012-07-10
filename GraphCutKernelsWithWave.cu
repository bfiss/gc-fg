/*
 * GraphCutKernels.cu
 *
 *  Created on: Jun 8, 2012
 *      Author: bruno
 */

#ifndef GRAPHCUTKERNELS_CU_
#define GRAPHCUTKERNELS_CU_

#include <stdio.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
//#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

#define DO_PUSH_C(edge,edge_inv,x_min,x_max,y_min,y_max,x_gap,y_gap,comp_h_idx)                                          \
	do{                                                                                              \
		excess = local_excess[local_idx];     \
		cap = edge[thread_id];                                                                       \
		bool changed = false; \
		if(cap > 0 && excess > 0 && /*local_height[local_idx] == */                                      \
           /*local_height[local_idx + (x_gap) + (y_gap) * 34] + 1*/ (comp_h&(1<<(comp_h_idx)))  ) {                                   \
			flow = excess > cap ? cap : excess;                                                      \
			/*excess -= flow;                                                                          \*/ \
			edge[thread_id] -= flow;                                                                 \
			edge_inv[thread_id + (x_gap) + (y_gap) * k.g.width_ex] += flow;                              \
			/*atomicSub(&k.g.n.excess[thread_id], flow);                                               \*/ \
			/*atomicAdd(&k.g.n.excess[thread_id + (x_gap) + (y_gap) * k.g.width_ex], flow);            \*/ \
			local_excess[local_idx] -= flow; \
			did_something = changed = true;                                                                    \
		}  \
		__syncthreads(); \
		changed ? local_excess[local_idx + (x_gap) + (y_gap) * 34] += flow : 0; \
		__syncthreads(); \
	} while(0)

#define DO_PUSH(edge,edge_inv,x_min,x_max,y_min,y_max,comp_h_idx) DO_PUSH_C(edge,edge_inv,x_min,x_max,y_min,y_max,(x_max-x_min),(y_max-y_min),comp_h_idx)

__global__ void InitGraph(KernelWrapper k, int * data_positive, int * data_negative, int penalty) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_id = x + y * k.g.width_ex;
	int block_id = blockIdx.x + blockIdx.y * k.block_x;
	k.active[block_id] = 0;

	__syncthreads();

	bool inside_area = x > 0 && y > 0 && x < k.g.width - 1 && y < k.g.height - 1;

	inside_area ? k.active[block_id] = 1 : 0;
	int coming = inside_area ? data_positive[x + y * k.g.width] : 0;
	int going = inside_area ? data_negative[x + y * k.g.width] : 0;

	k.g.n.edge_u[thread_id] = inside_area ? penalty : 0;
	k.g.n.edge_d[thread_id] = inside_area ? penalty : 0;
	k.g.n.edge_l[thread_id] = inside_area ? penalty : 0;
	k.g.n.edge_r[thread_id] = inside_area ? penalty : 0;
#if NEIGHBORHOOD == 8
	k.g.n.edge_ul[thread_id] = inside_area ? penalty : 0;
	k.g.n.edge_ur[thread_id] = inside_area ? penalty : 0;
	k.g.n.edge_dl[thread_id] = inside_area ? penalty : 0;
	k.g.n.edge_dr[thread_id] = inside_area ? penalty : 0;
#endif
	k.g.n.height[thread_id] = going >= coming ? 1 : 2;
	k.g.n.excess[thread_id] = coming - going;
	k.g.n.status[thread_id] = going >= coming ? 0 : 1;
	
	x == 1             ? k.g.n.edge_l[thread_id] = 0 : 0;
	x == k.g.width - 2 ? k.g.n.edge_r[thread_id] = 0 : 0;
	y == 1             ? k.g.n.edge_u[thread_id] = 0 : 0;
	y == k.g.height - 2 ? k.g.n.edge_d[thread_id] = 0 : 0;
#if NEIGHBORHOOD == 8
	x == 1             ? k.g.n.edge_ul[thread_id] = k.g.n.edge_dl[thread_id] = 0 : 0;
	x == k.g.width - 2 ? k.g.n.edge_ur[thread_id] = k.g.n.edge_dr[thread_id] = 0 : 0;
	y == 1             ? k.g.n.edge_ul[thread_id] = k.g.n.edge_ur[thread_id] = 0 : 0;
	y == k.g.height - 2 ? k.g.n.edge_dr[thread_id] = k.g.n.edge_dl[thread_id] = 0 : 0;
#endif
}

#ifdef SPREAD_ZEROS
#define UPDATE_COMP_N_LABEL(i,edge) comp_n |= (1<<(i)) * (!edge[thread_id])
#endif

__global__ void InitLabels(KernelWrapper k, int * label) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_id = x + y * k.g.width_ex;
	int label_id = x + y * k.g.width;
	
	if (x > 0 && y > 0 && x < k.g.width - 1 && y < k.g.height - 1) {
		//label[label_id] = k.g.n.height[thread_id] > k.g.size;
		//label[label_id] = k.g.n.excess[thread_id] > 0;
#ifdef SPREAD_ZEROS
		label[label_id] = k.g.n.excess[thread_id] >= 0;
#else
		label[label_id] = k.g.n.excess[thread_id] > 0;
#endif
	} else if(x < k.g.width && y < k.g.height) {
		label[label_id] = 0;
	}

	int comp_n = 0;
	UPDATE_COMP_N_LABEL(0,k.g.n.edge_d);
	UPDATE_COMP_N_LABEL(1,k.g.n.edge_u);
	UPDATE_COMP_N_LABEL(2,k.g.n.edge_r);
	UPDATE_COMP_N_LABEL(3,k.g.n.edge_l);
#if NEIGHBORHOOD == 8
	UPDATE_COMP_N_LABEL(4,k.g.n.edge_dr);
	UPDATE_COMP_N_LABEL(5,k.g.n.edge_dl);
	UPDATE_COMP_N_LABEL(6,k.g.n.edge_ur);
	UPDATE_COMP_N_LABEL(7,k.g.n.edge_ul);
#endif
	k.g.n.comp_n[thread_id] = comp_n;
}

__global__ void SpreadLabels(KernelWrapper k, int * label, int * alive) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_id = x + y * k.g.width_ex;
	int label_id = x + y * k.g.width;
	int local_idx = (threadIdx.y + 1) * 34 + threadIdx.x + 1;
	
	int comp_n = k.g.n.comp_n[thread_id];

	__shared__ int local_label[356];
	
	if (x < k.g.width && y < k.g.height) {

		local_label[local_idx] = label[label_id];

		threadIdx.x == 31 && x < k.g.width - 1 ? local_label[local_idx + 1]
				= label[label_id + 1] : 0;
		threadIdx.x == 0 && x > 0 ? local_label[local_idx - 1]
				= label[label_id - 1] : 0;
		threadIdx.y == 7 && y < k.g.height - 1 ? local_label[local_idx + 34]
				= label[label_id + k.g.width] : 0;
		threadIdx.y == 0 && y > 0 ? local_label[local_idx - 34]
				= label[label_id - k.g.width] : 0;

#if NEIGHBORHOOD == 8
		threadIdx.x == 0  && threadIdx.y == 0 &&
		x > 0 && y > 0 ? local_label[local_idx - 35] = label[label_id - 1 - k.g.width] : 0;
		threadIdx.x == 31 && threadIdx.y == 0 &&
		x < k.g.width - 1 && y > 0 ? local_label[local_idx - 33] = label[label_id + 1 - k.g.width] : 0;
		threadIdx.x == 0  && threadIdx.y == 7 &&
		x > 0 && y < k.g.height - 1 ? local_label[local_idx + 33] = label[label_id - 1 + k.g.width] : 0;
		threadIdx.x == 31 && threadIdx.y == 7 &&
		x < k.g.width - 1 && y < k.g.height - 1 ? local_label[local_idx + 35] = label[label_id + 1 + k.g.width] : 0;
#endif
	}
	
	__syncthreads();

	int curr_label = local_label[local_idx];
	int orig_label = curr_label;
	
	int repetitions = 4;

	do{
		if (x > 0 && y > 0 && x < k.g.width - 1 && y < k.g.height - 1) {

#ifdef SPREAD_ZEROS
			curr_label = ( (comp_n & (1<<0)) || local_label[local_idx+34]) && curr_label;
			curr_label = ( (comp_n & (1<<1)) || local_label[local_idx-34]) && curr_label;
			curr_label = ( (comp_n & (1<<2)) || local_label[local_idx+1]) && curr_label;
			curr_label = ( (comp_n & (1<<3)) || local_label[local_idx-1]) && curr_label;
	#if NEIGHBORHOOD == 8
			curr_label = ( (comp_n & (1<<4)) || local_label[local_idx+35]) && curr_label;
			curr_label = ( (comp_n & (1<<5)) || local_label[local_idx+33]) && curr_label;
			curr_label = ( (comp_n & (1<<6)) || local_label[local_idx-33]) && curr_label;
			curr_label = ( (comp_n & (1<<7)) || local_label[local_idx-35]) && curr_label;
	#endif
#else
			curr_label = k.g.n.edge_u[thread_id+k.g.width_ex] && local_label[local_idx+34] || curr_label;
			curr_label = k.g.n.edge_d[thread_id-k.g.width_ex] && local_label[local_idx-34] || curr_label;
			curr_label = k.g.n.edge_l[thread_id+1] && local_label[local_idx+1] || curr_label;
			curr_label = k.g.n.edge_r[thread_id-1] && local_label[local_idx-1] || curr_label;
	#if NEIGHBORHOOD == 8
			curr_label = k.g.n.edge_ul[thread_id+k.g.width_ex+1] && local_label[local_idx+35] || curr_label;
			curr_label = k.g.n.edge_ur[thread_id+k.g.width_ex-1] && local_label[local_idx+33] || curr_label;
			curr_label = k.g.n.edge_dl[thread_id-k.g.width_ex+1] && local_label[local_idx-33] || curr_label;
			curr_label = k.g.n.edge_dr[thread_id-k.g.width_ex-1] && local_label[local_idx-35] || curr_label;
	#endif
#endif
			curr_label != orig_label ? label[label_id] = curr_label : 0;
			curr_label != orig_label ? alive[0] = 1 : 0;
			orig_label = curr_label;
		}
		__syncthreads();
	} while(--repetitions);
}

__global__ void Push(KernelWrapper k, int iter, int skip, int * alive) {
	if (!skip || k.active[blockIdx.x + blockIdx.y * k.block_x]) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int thread_id = x + y * k.g.width_ex;

		int comp_h = k.g.n.comp_h[thread_id];

		int local_idx = (threadIdx.y + 1) * 34 + threadIdx.x + 1;

		/*__shared__ int local_height[356];

		local_height[local_idx] = k.g.n.height[thread_id];

		threadIdx.x == 31 && x < k.g.width_ex - 1 ? local_height[local_idx + 1]
				= k.g.n.height[thread_id + 1] : 0;
		threadIdx.x == 0 && x > 0 ? local_height[local_idx - 1]
				= k.g.n.height[thread_id - 1] : 0;
		threadIdx.y == 7 && y < k.g.height_ex - 1 ? local_height[local_idx + 34]
		        = k.g.n.height[thread_id + k.g.width_ex] : 0;
		threadIdx.y == 0 && y > 0 ? local_height[local_idx - 34]
				= k.g.n.height[thread_id - k.g.width_ex] : 0;

#if NEIGHBORHOOD == 8
		threadIdx.x == 0 && threadIdx.y == 0 &&
		x > 0 && y > 0 ? local_height[local_idx - 35] = k.g.n.height[thread_id - 1 - k.g.width_ex] : 0;
		threadIdx.x == 31 && threadIdx.y == 0 &&
		x < k.g.width_ex - 1 && y > 0 ? local_height[local_idx - 33] = k.g.n.height[thread_id + 1 - k.g.width_ex] : 0;
		threadIdx.x == 0 && threadIdx.y == 7 &&
		x > 0 && y < k.g.height_ex - 1 ? local_height[local_idx + 33] = k.g.n.height[thread_id - 1 + k.g.width_ex] : 0;
		threadIdx.x == 31 && threadIdx.y == 7 &&
		x < k.g.width_ex - 1 && y < k.g.height_ex - 1 ? local_height[local_idx + 35] = k.g.n.height[thread_id + 1 + k.g.width_ex] : 0;
#endif*/

		__shared__ int local_excess[356];

		local_excess[local_idx] = k.g.n.excess[thread_id];

		threadIdx.x == 31 && x < k.g.width_ex - 1  ? local_excess[local_idx +  1] = 0 : 0;
		threadIdx.x == 0  && x > 0                 ? local_excess[local_idx -  1] = 0 : 0;
		threadIdx.y == 7  && y < k.g.height_ex - 1 ? local_excess[local_idx + 34] = 0 : 0;
		threadIdx.y == 0  && y > 0                 ? local_excess[local_idx - 34] = 0 : 0;

#if NEIGHBORHOOD == 8
		threadIdx.x == 0 && threadIdx.y == 0 && x > 0                && y > 0                 ? local_excess[local_idx -35] = 0 : 0;
		threadIdx.x == 31&& threadIdx.y == 0 && x < k.g.width_ex - 1 && y > 0                 ? local_excess[local_idx -33] = 0 : 0;
		threadIdx.x == 0 && threadIdx.y == 7 && x > 0                && y < k.g.height_ex - 1 ? local_excess[local_idx +33] = 0 : 0;
		threadIdx.x == 31&& threadIdx.y == 7 && x < k.g.width_ex - 1 && y < k.g.height_ex - 1 ? local_excess[local_idx +35] = 0 : 0;
#endif

		__syncthreads();
		
		bool did_something = false;

		int original_excess = local_excess[local_idx];

		//if (k.g.n.status[thread_id]) {
			int excess;
			int cap;
			int flow;
			do {

				DO_PUSH(k.g.n.edge_l,k.g.n.edge_r,1,0,0,0,0);
				DO_PUSH(k.g.n.edge_r,k.g.n.edge_l,0,1,0,0,1);
				DO_PUSH(k.g.n.edge_u,k.g.n.edge_d,0,0,1,0,2);
				DO_PUSH(k.g.n.edge_d,k.g.n.edge_u,0,0,0,1,3);
#if NEIGHBORHOOD == 8
				DO_PUSH(k.g.n.edge_dr,k.g.n.edge_ul,0,1,0,1,4);
				DO_PUSH(k.g.n.edge_dl,k.g.n.edge_ur,1,0,0,1,5);
				DO_PUSH(k.g.n.edge_ur,k.g.n.edge_dl,0,1,1,0,6);
				DO_PUSH(k.g.n.edge_ul,k.g.n.edge_dr,1,0,1,0,7);
#endif

				//excess = k.g.n.excess[thread_id];
			} while (--iter);
			excess = local_excess[local_idx];

			/*if(threadIdx.x > 0 && threadIdx.x < 31 && threadIdx.y > 0 && threadIdx.y < 7)
				excess - original_excess ? k.g.n.excess[thread_id] = excess : 0;
			else*/
				excess - original_excess ? atomicAdd(&k.g.n.excess[thread_id], excess - original_excess) : 0;
				
			threadIdx.x == 0 && x > 0 && local_excess[local_idx - 1] ? atomicAdd(&k.g.n.excess[thread_id - 1], local_excess[local_idx - 1]) : 0;
			threadIdx.y == 0 && y > 0 && local_excess[local_idx - 34] ? atomicAdd(&k.g.n.excess[thread_id - k.g.width_ex], local_excess[local_idx - 34]) : 0;
			threadIdx.x == 31 && x < k.g.width_ex - 1 && local_excess[local_idx + 1] ? atomicAdd(&k.g.n.excess[thread_id + 1], local_excess[local_idx + 1]) : 0;
			threadIdx.y == 7 && y < k.g.height_ex - 1 && local_excess[local_idx + 34] ? atomicAdd(&k.g.n.excess[thread_id + k.g.width_ex], local_excess[local_idx + 34]) : 0;
#if NEIGHBORHOOD == 8
#endif
		//}
		if(!skip && did_something)
			alive[0] = 1;
	}
}

#define ADJUST_HEIGHT(diff,edge) (height > local_height[local_idx + (diff)] && (edge)[thread_id] > 0) ? height = local_height[local_idx + (diff)] : 0

#define UPDATE_COMP_H(i,diff) comp_h |= (1 << (i)) * (local_height[local_idx] == local_height[local_idx+(diff)] + 1)

__global__ void Relabel(KernelWrapper k, int skip) {
	if (!skip || k.active[blockIdx.x + blockIdx.y * k.block_x]) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int thread_id = x + y * k.g.width_ex;

		__shared__ int local_height[356];

		int local_idx = (threadIdx.y + 1) * 34 + threadIdx.x + 1;

		local_height[local_idx] = k.g.n.height[thread_id];

		threadIdx.x == 31 && x < k.g.width_ex - 1 ? local_height[local_idx + 1]
				= k.g.n.height[thread_id + 1] : 0;
		threadIdx.x == 0 && x > 0 ? local_height[local_idx - 1]
				= k.g.n.height[thread_id - 1] : 0;
		threadIdx.y == 7 && y < k.g.height_ex - 1 ? local_height[local_idx + 34]
		        = k.g.n.height[thread_id + k.g.width_ex] : 0;
		threadIdx.y == 0 && y > 0 ? local_height[local_idx - 34]
				= k.g.n.height[thread_id - k.g.width_ex] : 0;

		#if NEIGHBORHOOD == 8
		threadIdx.x == 0 && threadIdx.y == 0 &&
		x > 0 && y > 0 ? local_height[local_idx - 35] = k.g.n.height[thread_id - 1 - k.g.width_ex] : 0;
		threadIdx.x == 31 && threadIdx.y == 0 &&
		x < k.g.width_ex - 1 && y > 0 ? local_height[local_idx - 33] = k.g.n.height[thread_id + 1 - k.g.width_ex] : 0;
		threadIdx.x == 0 && threadIdx.y == 7 &&
		x > 0 && y < k.g.height_ex - 1 ? local_height[local_idx + 33] = k.g.n.height[thread_id - 1 + k.g.width_ex] : 0;
		threadIdx.x == 31 && threadIdx.y == 7 &&
		x < k.g.width_ex - 1 && y < k.g.height_ex - 1 ? local_height[local_idx + 35] = k.g.n.height[thread_id + 1 + k.g.width_ex] : 0;
		#endif
		
		__syncthreads();
		
		int excess = k.g.n.excess[thread_id];
		//int status = 0;

		if (excess >= 0 && x > 0 && y > 0 && x < k.g.width - 1 && y < k.g.height - 1) {
			int height = k.g.size;

			ADJUST_HEIGHT( -1,k.g.n.edge_l);
			ADJUST_HEIGHT(  1,k.g.n.edge_r);
			ADJUST_HEIGHT(-34,k.g.n.edge_u);
			ADJUST_HEIGHT( 34,k.g.n.edge_d);
#if NEIGHBORHOOD == 8
			ADJUST_HEIGHT(-35,k.g.n.edge_ul);
			ADJUST_HEIGHT(-33,k.g.n.edge_ur);
			ADJUST_HEIGHT( 33,k.g.n.edge_dl);
			ADJUST_HEIGHT( 35,k.g.n.edge_dr);
#endif
			//height != k.g.size_ex ? printf("Changed from %d to %d\n",k.g.n.height[thread_id],height): 0;
			//status = height != k.g.size;
			k.g.n.height[thread_id] = height + 1;
			//__sync?
			local_height[local_idx] = height + 1;
		}

		__syncthreads();
		int comp_h = 0;
		if (x > 0 && y > 0 && x < k.g.width - 1 && y < k.g.height - 1) {
			UPDATE_COMP_H(0, -1);
			UPDATE_COMP_H(1,  1);
			UPDATE_COMP_H(2,-34);
			UPDATE_COMP_H(3, 34);
	#if NEIGHBORHOOD == 8
			UPDATE_COMP_H(7,-35);
			UPDATE_COMP_H(6,-33);
			UPDATE_COMP_H(5, 33);
			UPDATE_COMP_H(4, 35);
	#endif
		}
		k.g.n.comp_h[thread_id] = comp_h;

		k.g.n.status[thread_id] = excess > 0 && local_height[local_idx] != k.g.size + 1;
		//k.g.n.status[thread_id] = status;
	}
}

__global__ void UpdateActivity(int * status, int * active, int block_x, int width_ex) {
	int block_id = blockIdx.x + blockIdx.y * block_x;
	active[block_id] = 0;

	__syncthreads();

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_id = x + y * width_ex;

	status[thread_id] ? active[block_id] = 1 : 0;
}

#define UPDATE_COMP_N(i,edge) comp_n |= (1<<(i)) * (edge[thread_id] > 0)

__global__ void InitGlobalRelabel(KernelWrapper k) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_id = x + y * k.g.width_ex;

	int no_sink = 0;
	if (x > 0 && y > 0 && x < k.g.width - 1 && y < k.g.height - 1) {
		no_sink = k.g.n.excess[thread_id] >= 0;
		k.g.n.height[thread_id] = no_sink * k.g.size + 1;
	}

	int comp_n = 0;
	UPDATE_COMP_N(0,k.g.n.edge_l);
	UPDATE_COMP_N(1,k.g.n.edge_r);
	UPDATE_COMP_N(2,k.g.n.edge_u);
	UPDATE_COMP_N(3,k.g.n.edge_d);
#if NEIGHBORHOOD == 8
	UPDATE_COMP_N(4,k.g.n.edge_ul);
	UPDATE_COMP_N(5,k.g.n.edge_ur);
	UPDATE_COMP_N(6,k.g.n.edge_dl);
	UPDATE_COMP_N(7,k.g.n.edge_dr);
#endif
	/* different because of >= instead of > */
	/* and accounting for activity instead of only no-sinkness */
	comp_n |= (1<<8) * no_sink;
	k.g.n.comp_n[thread_id] = comp_n;
}

#define COMP_ADJUST_HEIGHT(diff,i) (height > local_height[local_idx + (diff)] && ((1<<(i)) & comp_n)) ? height = local_height[local_idx + (diff)] : 0

__global__ void GlobalRelabel(KernelWrapper k, int * alive) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_id = x + y * k.g.width_ex;

	int comp_n = k.g.n.comp_n[thread_id];
	if ( __syncthreads_or(comp_n & (1<<8)) ) {

		__shared__ int local_height[356];

		int local_idx = (threadIdx.y + 1) * 34 + threadIdx.x + 1;

		local_height[local_idx] = k.g.n.height[thread_id];

		/*int outer_repetitions = 1;

		do {*/
			threadIdx.x == 31 && x < k.g.width_ex - 1 ?
					local_height[local_idx + 1] = k.g.n.height[thread_id + 1] : 0;
			threadIdx.x == 0 && x > 0 ?
					local_height[local_idx - 1] = k.g.n.height[thread_id - 1] : 0;
			threadIdx.y == 7 && y < k.g.height_ex - 1 ?
					local_height[local_idx + 34] =
							k.g.n.height[thread_id + k.g.width_ex] :
					0;
			threadIdx.y == 0 && y > 0 ?
					local_height[local_idx - 34] =
							k.g.n.height[thread_id - k.g.width_ex] :
					0;

		#if NEIGHBORHOOD == 8
			threadIdx.x == 0 && threadIdx.y == 0 &&
			x > 0 && y > 0 ? local_height[local_idx - 35] = k.g.n.height[thread_id - 1 - k.g.width_ex] : 0;
			threadIdx.x == 31 && threadIdx.y == 0 &&
			x < k.g.width_ex - 1 && y > 0 ? local_height[local_idx - 33] = k.g.n.height[thread_id + 1 - k.g.width_ex] : 0;
			threadIdx.x == 0 && threadIdx.y == 7 &&
			x > 0 && y < k.g.height_ex - 1 ? local_height[local_idx + 33] = k.g.n.height[thread_id - 1 + k.g.width_ex] : 0;
			threadIdx.x == 31 && threadIdx.y == 7 &&
			x < k.g.width_ex - 1 && y < k.g.height_ex - 1 ? local_height[local_idx + 35] = k.g.n.height[thread_id + 1 + k.g.width_ex] : 0;
		#endif

			__syncthreads();

				bool changed = false;

				int repetitions = 10;
				do {
					int height = local_height[local_idx] - 1;
					if (((1<<8) & comp_n) && x > 0 && y > 0	&& x < k.g.width - 1 && y < k.g.height - 1) {
						height = k.g.size;

						COMP_ADJUST_HEIGHT( -1, 0);
						COMP_ADJUST_HEIGHT(  1, 1);
						COMP_ADJUST_HEIGHT(-34, 2);
						COMP_ADJUST_HEIGHT( 34, 3);
			#if NEIGHBORHOOD == 8
						COMP_ADJUST_HEIGHT(-35, 4);
						COMP_ADJUST_HEIGHT(-33, 5);
						COMP_ADJUST_HEIGHT( 33, 6);
						COMP_ADJUST_HEIGHT( 35, 7);
			#endif
					}
					__syncthreads();
					changed |= (local_height[local_idx] != height + 1);
					local_height[local_idx] = height + 1;
					__syncthreads();
				} while (--repetitions);
				//height != k.g.size_ex ? printf("Changed from %d to %d\n",k.g.n.height[thread_id],height): 0;
				changed ? k.g.n.comp_n[thread_id] = comp_n ^ (1<<8) : 0;
				changed ? k.g.n.height[thread_id] = local_height[local_idx] : 0;
				//k.g.n.height[thread_id] = height + 1;

				//__syncthreads();

				changed ? alive[0] = 1 : 0;
		//} while(--outer_repetitions);
	}
}

#endif /* GRAPHCUTKERNELS_CU_ */
