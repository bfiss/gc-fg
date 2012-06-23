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

#define DO_PUSH_C(edge,x_min,x_max,y_min,y_max,x_gap,y_gap)                                          \
	do{                                                                                              \
		cap = edge[thread_id];                                                                       \
		if(cap > 0 && excess > 0 && local_height[local_idx] ==                                       \
           local_height[local_idx + (x_gap) + (y_gap) * 34] + 1) {                                   \
			flow = excess > cap ? cap : excess;                                                      \
			excess -= flow;                                                                          \
			edge[thread_id] -= flow;                                                                 \
			edge[thread_id + (x_gap) + (y_gap) * k.g.width_ex] += flow;                              \
			atomicSub(&k.g.n.excess[thread_id], flow);                                               \
			atomicAdd(&k.g.n.excess[thread_id + (x_gap) + (y_gap) * k.g.width_ex], flow);            \
			did_something = true;                                                                    \
		}                                                                                            \
	} while(0)

#define DO_PUSH(edge,x_min,x_max,y_min,y_max) DO_PUSH_C(edge,x_min,x_max,y_min,y_max,(x_max-x_min),(y_max-y_min))

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

	k.g.n.edge_sink[thread_id] = going >= coming ? going - coming : 0;
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
	k.g.n.excess[thread_id] = going >= coming ? 0 : coming - going;
	k.g.n.status[thread_id] = going >= coming ? 0 : 1;
	
	x == 1             ? k.g.n.edge_l[thread_id] = 0 : 0;
	x == k.g.width - 2 ? k.g.n.edge_r[thread_id] = 0 : 0;
	y == 1             ? k.g.n.edge_u[thread_id] = 0 : 0;
	y == k.g.width - 2 ? k.g.n.edge_d[thread_id] = 0 : 0;
#if NEIGHBORHOOD == 8
	x == 1             ? k.g.n.edge_ul[thread_id] = k.g.n.edge_dl[thread_id] = 0 : 0;
	x == k.g.width - 2 ? k.g.n.edge_ur[thread_id] = k.g.n.edge_dr[thread_id] = 0 : 0;
	y == 1             ? k.g.n.edge_ul[thread_id] = k.g.n.edge_ur[thread_id] = 0 : 0;
	y == k.g.width - 2 ? k.g.n.edge_dr[thread_id] = k.g.n.edge_dl[thread_id] = 0 : 0;
#endif
}

__global__ void InitLabels(KernelWrapper k, int * label) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_id = x + y * k.g.width_ex;
	int label_id = x + y * k.g.width;
	
	if (x > 0 && y > 0 && x < k.g.width - 1 && y < k.g.height - 1) {
		//label[label_id] = k.g.n.height[thread_id] >= 15;
		//label[label_id] = k.g.n.excess[thread_id] > 0;
		label[label_id] = k.g.n.edge_sink[thread_id] == 0;
	} else if(x < k.g.width && y < k.g.height) {
		label[label_id] = 0;
	}
}

__global__ void SpreadLabels(KernelWrapper k, int * label, int * alive) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_id = x + y * k.g.width_ex;
	int label_id = x + y * k.g.width;
	int local_idx = (threadIdx.y + 1) * 34 + threadIdx.x + 1;
	
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
	
	if (x > 0 && y > 0 && x < k.g.width - 1 && y < k.g.height - 1) {
		int curr_label = local_label[local_idx];
		int orig_label = curr_label;

/*      Spreading ones:		
		curr_label = k.g.n.edge_u[thread_id+k.g.width_ex] && local_label[local_idx+34] || curr_label;
		curr_label = k.g.n.edge_d[thread_id-k.g.width_ex] && local_label[local_idx-34] || curr_label;
		curr_label = k.g.n.edge_l[thread_id+1] && local_label[local_idx+1] || curr_label;
		curr_label = k.g.n.edge_r[thread_id-1] && local_label[local_idx-1] || curr_label;
#if NEIGHBORHOOD == 8
		curr_label = k.g.n.edge_ul[thread_id+k.g.width_ex+1] && local_label[local_idx+35] || curr_label;
		curr_label = k.g.n.edge_ur[thread_id+k.g.width_ex-1] && local_label[local_idx+33] || curr_label;
		curr_label = k.g.n.edge_dl[thread_id-k.g.width_ex+1] && local_label[local_idx-33] || curr_label;
		curr_label = k.g.n.edge_dr[thread_id-k.g.width_ex-1] && local_label[local_idx-35] || curr_label;
#endif*/

/*		Spreading zeroes: */
		curr_label = (!k.g.n.edge_d[thread_id] || local_label[local_idx+34]) && curr_label;
		curr_label = (!k.g.n.edge_u[thread_id] || local_label[local_idx-34]) && curr_label;
		curr_label = (!k.g.n.edge_r[thread_id] || local_label[local_idx+1]) && curr_label;
		curr_label = (!k.g.n.edge_l[thread_id] || local_label[local_idx-1]) && curr_label;
#if NEIGHBORHOOD == 8
		curr_label = (!k.g.n.edge_dr[thread_id] || local_label[local_idx+35]) && curr_label;
		curr_label = (!k.g.n.edge_dl[thread_id] || local_label[local_idx+33]) && curr_label;
		curr_label = (!k.g.n.edge_ur[thread_id] || local_label[local_idx-33]) && curr_label;
		curr_label = (!k.g.n.edge_ul[thread_id] || local_label[local_idx-35]) && curr_label;
#endif

		label[label_id] = curr_label;
		curr_label != orig_label ? alive[0] = 1 : 0;
	}
}

__global__ void Push(KernelWrapper k, int iter, int skip, int * alive) {
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
		
		bool did_something = false;

		if (k.g.n.status[thread_id]) {
			int excess = k.g.n.excess[thread_id];
			int cap;
			int flow;
			do {
				cap = k.g.n.edge_sink[thread_id];
				if (cap > 0 && excess > 0 && local_height[local_idx] == 1) {
					flow = excess > cap ? cap : excess;
					excess -= flow;
					k.g.n.edge_sink[thread_id] -= flow;
					atomicSub(&k.g.n.excess[thread_id], flow);
					did_something = true;
				}

				DO_PUSH(k.g.n.edge_u,0,0,1,0);
				DO_PUSH(k.g.n.edge_d,0,0,0,1);
				DO_PUSH(k.g.n.edge_l,1,0,0,0);
				DO_PUSH(k.g.n.edge_r,0,1,0,0);
#if NEIGHBORHOOD == 8
				DO_PUSH(k.g.n.edge_ul,1,0,1,0);
				DO_PUSH(k.g.n.edge_ur,0,1,1,0);
				DO_PUSH(k.g.n.edge_dl,1,0,0,1);
				DO_PUSH(k.g.n.edge_dr,0,1,0,1);
#endif

				excess = k.g.n.excess[thread_id];
			} while (--iter);
		}
		if(!skip && did_something)
			alive[0] = 1;
	}
}

#define ADJUST_HEIGHT(diff,edge) (height > local_height[local_idx + (diff)] && (edge)[thread_id] > 0) ? height = local_height[local_idx + (diff)] : 0

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
		int status = k.g.n.edge_sink[thread_id] != 0 && excess > 0;

		if (excess > 0 && !status && x > 0 && y > 0 && x < k.g.width - 1 && y < k.g.height - 1) {
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
			status = height != k.g.size;
			k.g.n.height[thread_id] = height + 1;
		}
		k.g.n.status[thread_id] = status;
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

#endif /* GRAPHCUTKERNELS_CU_ */
