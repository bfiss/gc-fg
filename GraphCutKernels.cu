/*!
 * \file GraphCutKernels.cu
 *
 * \author bruno
 * \date Jun 8, 2012
 *
 * CUDA source file that contains the kernels for this Graph Cut implementation.
 */

#ifndef GRAPHCUTKERNELS_CU_
#define GRAPHCUTKERNELS_CU_

#include <stdio.h>

/*! \def SHARED_MEMORY_SIZE
 * \brief The size of shared memory vectors to hold height or excess information.
 *
 * The size of shared memory vectors to hold height or excess information.
 */
#define SHARED_MEMORY_SIZE ((THREADS_X + 2) * (THREADS_Y + 2))

/*! \def DO_PUSH_C(edge,edge_inv,x_min,x_max,y_min,y_max,x_gap,y_gap,comp_h_idx)
 * \brief Internal macro used to make a push in a certain direction with a certain node.
 */
#define DO_PUSH_C(edge,edge_inv,x_min,x_max,y_min,y_max,x_gap,y_gap,comp_h_idx)                     \
	do{                                                                                              \
		cap = edge[thread_id];                                                                       \
		if(cap > 0 && excess > 0 && local_height[local_idx] ==                                       \
           local_height[local_idx + (x_gap) + (y_gap) * (THREADS_X + 2)] + 1 /*(comp_h&(1<<(comp_h_idx)))*/  ) {          \
			flow = excess > cap ? cap : excess;                                                      \
			excess -= flow;                                                                          \
			edge[thread_id] -= flow;                                                                 \
			edge_inv[thread_id + (x_gap) + (y_gap) * k.g.width_ex] += flow;                          \
			atomicSub(&k.g.n.excess[thread_id], flow);                                               \
			atomicAdd(&k.g.n.excess[thread_id + (x_gap) + (y_gap) * k.g.width_ex], flow);            \
			did_something = true;                                                                    \
		}                                                                                            \
	} while(0)

/*! \def DO_PUSH(edge,edge_inv,x_min,x_max,y_min,y_max,comp_h_idx)
 * \brief Short version for DO_PUSH_C macro.
 */
#define DO_PUSH(edge,edge_inv,x_min,x_max,y_min,y_max,comp_h_idx) DO_PUSH_C(edge,edge_inv,x_min,x_max,y_min,y_max,(x_max-x_min),(y_max-y_min),comp_h_idx)

/*! \def DO_PUSH_C_WAVE(edge,edge_inv,x_min,x_max,y_min,y_max,x_gap,y_gap,comp_h_idx)
 * \brief Internal macro used to make a push in a certain direction with a certain node.
 *
 * This version does not use atomic functions, but synchronizes. An attempt to improve the wave operator
 * by Timo Stich, but that is not very successful, perhaps due to too frequent synchronization.
 */
#define DO_PUSH_C_WAVE(edge,edge_inv,x_min,x_max,y_min,y_max,x_gap,y_gap,comp_h_idx)                \
	do{                                                                                              \
		excess = local_excess[local_idx];                                                            \
		cap = edge[thread_id];                                                                       \
		bool changed = false;                                                                       \
		if(cap > 0 && excess > 0 && local_height[local_idx] ==                                       \
           local_height[local_idx + (x_gap) + (y_gap) * (THREADS_X + 2)] + 1 /*(comp_h&(1<<(comp_h_idx)))*/  ) {     \
			flow = excess > cap ? cap : excess;                                                      \
			/*excess -= flow;                                                                    \*/ \
			edge[thread_id] -= flow;                                                                 \
			edge_inv[thread_id + (x_gap) + (y_gap) * k.g.width_ex] += flow;                          \
			/*atomicSub(&k.g.n.excess[thread_id], flow);                                         \*/ \
			/*atomicAdd(&k.g.n.excess[thread_id + (x_gap) + (y_gap) * k.g.width_ex], flow);      \*/ \
			local_excess[local_idx] -= flow;                                                         \
			did_something = changed = true;                                                          \
		}  \
		__syncthreads(); \
		changed ? local_excess[local_idx + (x_gap) + (y_gap) * (THREADS_X + 2)] += flow : 0; \
		__syncthreads(); \
	} while(0)

/*! \def DO_PUSH_WAVE(edge,edge_inv,x_min,x_max,y_min,y_max,comp_h_idx)
 * \brief Short version for DO_PUSH_C_WAVE macro.
 */
#define DO_PUSH_WAVE(edge,edge_inv,x_min,x_max,y_min,y_max,comp_h_idx) DO_PUSH_C_WAVE(edge,edge_inv,x_min,x_max,y_min,y_max,(x_max-x_min),(y_max-y_min),comp_h_idx)

/*! \def DO_PUSH_C_WAVE2(edge,edge_inv,x_min,x_max,y_min,y_max,x_gap,y_gap,comp_h_idx)
 * \brief Internal macro used to make a push in a certain direction with a certain node.
 *
 * This version uses atomic functions and shared memory for flow. An attempt to improve the wave operator
 * by Timo Stich.
 */
#define DO_PUSH_C_WAVE2(edge,edge_inv,x_min,x_max,y_min,y_max,x_gap,y_gap,comp_h_idx)               \
	do{                                                                                              \
		excess = local_excess[local_idx];                                                            \
		__syncthreads(); /* only for determinism */                                                  \
		cap = edge[thread_id];                                                                       \
		if(cap > 0 && excess > 0 && local_height[local_idx] ==                                       \
           local_height[local_idx + (x_gap) + (y_gap) * (THREADS_X + 2)] + 1 /*(comp_h&(1<<(comp_h_idx)))*/  ) {                                   \
			flow = excess > cap ? cap : excess;                                                      \
			/*excess -= flow;                                                                    \*/ \
			edge[thread_id] -= flow;                                                                 \
			edge_inv[thread_id + (x_gap) + (y_gap) * k.g.width_ex] += flow;                          \
			atomicSub(&local_excess[local_idx], flow);                                               \
			atomicAdd(&local_excess[local_idx + (x_gap) + (y_gap) * (THREADS_X + 2)], flow);         \
			did_something = true;                                                                    \
		}  \
		__syncthreads(); \
	} while(0)

/*! \def DO_PUSH_WAVE2(edge,edge_inv,x_min,x_max,y_min,y_max,comp_h_idx)
 * \brief Short version for DO_PUSH_C_WAVE2 macro.
 */
#define DO_PUSH_WAVE2(edge,edge_inv,x_min,x_max,y_min,y_max,comp_h_idx) DO_PUSH_C_WAVE2(edge,edge_inv,x_min,x_max,y_min,y_max,(x_max-x_min),(y_max-y_min),comp_h_idx)

//! Initializes the internal structure of Graph Cut.
/*!
 * This kernel initializes the data structures used in the other kernels. It uses a fixed penalty as the edge capacity
 * between neighbors.
 *
 */
__global__ void InitGraph(KernelWrapper k, int * data_positive,
		int * data_negative, int penalty) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_id = x + y * k.g.width_ex;
	int block_id = blockIdx.x + blockIdx.y * k.block_x;
	k.active[block_id] = 0;

	__syncthreads();

	bool inside_area = x > 0 && y > 0 && x < k.g.width - 1
			&& y < k.g.height - 1;

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
	k.g.n.status[thread_id] = inside_area ? (going >= coming ? 2 : 1) : 0;

	x == 1 ? k.g.n.edge_l[thread_id] = 0 : 0;
	x == k.g.width - 2 ? k.g.n.edge_r[thread_id] = 0 : 0;
	y == 1 ? k.g.n.edge_u[thread_id] = 0 : 0;
	y == k.g.height - 2 ? k.g.n.edge_d[thread_id] = 0 : 0;
#if NEIGHBORHOOD == 8
	x == 1 ? k.g.n.edge_ul[thread_id] = k.g.n.edge_dl[thread_id] = 0 : 0;
	x == k.g.width - 2 ? k.g.n.edge_ur[thread_id] = k.g.n.edge_dr[thread_id] = 0 : 0;
	y == 1 ? k.g.n.edge_ul[thread_id] = k.g.n.edge_ur[thread_id] = 0 : 0;
	y == k.g.height - 2 ? k.g.n.edge_dr[thread_id] = k.g.n.edge_dl[thread_id] = 0 : 0;
#endif

	k.g.n.energy_sum[thread_id] = 0;
	k.g.n.comp_h[thread_id] = 0;
	k.g.n.comp_n[thread_id] = 0;
}

//! Initializes the internal structure of Graph Cut.
/*!
 * This kernel initializes the data structures used in the other kernels. It sets the edge capacities
 * between neighbors using the given parameters.
 *
 */
__global__ void InitGraphVarEdges(KernelWrapper k, int * data_positive,
		int * data_negative, int * up, int * down, int * left, int * right
#if NEIGHBORHOOD == 8
		, int * upleft, int * upright, int * downleft, int * downright
#endif
		) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_id = x + y * k.g.width_ex;
	int block_id = blockIdx.x + blockIdx.y * k.block_x;
	k.active[block_id] = 0;

	__syncthreads();

	bool inside_area = x > 0 && y > 0 && x < k.g.width - 1
			&& y < k.g.height - 1;

	inside_area ? k.active[block_id] = 1 : 0;
	int coming = inside_area ? data_positive[x + y * k.g.width] : 0;
	int going = inside_area ? data_negative[x + y * k.g.width] : 0;

	k.g.n.edge_u[thread_id] = inside_area ? up[x + y * k.g.width] : 0;
	k.g.n.edge_d[thread_id] = inside_area ? down[x + y * k.g.width] : 0;
	k.g.n.edge_l[thread_id] = inside_area ? left[x + y * k.g.width] : 0;
	k.g.n.edge_r[thread_id] = inside_area ? right[x + y * k.g.width] : 0;
#if NEIGHBORHOOD == 8
	k.g.n.edge_ul[thread_id] = inside_area ? upleft[x + y * k.g.width] : 0;
	k.g.n.edge_ur[thread_id] = inside_area ? upright[x + y * k.g.width] : 0;
	k.g.n.edge_dl[thread_id] = inside_area ? downleft[x + y * k.g.width] : 0;
	k.g.n.edge_dr[thread_id] = inside_area ? downright[x + y * k.g.width] : 0;
#endif
	k.g.n.height[thread_id] = going >= coming ? 1 : 2;
	k.g.n.excess[thread_id] = coming - going;
	k.g.n.status[thread_id] = inside_area ? (going >= coming ? 2 : 1) : 0;

	x == 1 ? k.g.n.edge_l[thread_id] = 0 : 0;
	x == k.g.width - 2 ? k.g.n.edge_r[thread_id] = 0 : 0;
	y == 1 ? k.g.n.edge_u[thread_id] = 0 : 0;
	y == k.g.height - 2 ? k.g.n.edge_d[thread_id] = 0 : 0;
#if NEIGHBORHOOD == 8
	x == 1 ? k.g.n.edge_ul[thread_id] = k.g.n.edge_dl[thread_id] = 0 : 0;
	x == k.g.width - 2 ? k.g.n.edge_ur[thread_id] = k.g.n.edge_dr[thread_id] = 0 : 0;
	y == 1 ? k.g.n.edge_ul[thread_id] = k.g.n.edge_ur[thread_id] = 0 : 0;
	y == k.g.height - 2 ? k.g.n.edge_dr[thread_id] = k.g.n.edge_dl[thread_id] = 0 : 0;
#endif

	k.g.n.energy_sum[thread_id] = 0;
	k.g.n.comp_h[thread_id] = 0;
	k.g.n.comp_n[thread_id] = 0;
}

#ifdef SPREAD_ZEROS
/*! \def UPDATE_COMP_N_LABEL(i,edge)
 * \brief Updates the compressed neighborhood for one direction.
 */
#define UPDATE_COMP_N_LABEL(i,edge) comp_n |= (1<<(i)) * (!edge[thread_id])
#endif

//! Initializes the final labeling
/*!
 * This kernel initializes \a label in order to spread values correctly in SpreadLabels.
 *
 */
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
	} else if (x < k.g.width && y < k.g.height) {
		label[label_id] = 0;
	}

	int comp_n = 0;
	UPDATE_COMP_N_LABEL(0, k.g.n.edge_d);
	UPDATE_COMP_N_LABEL(1, k.g.n.edge_u);
	UPDATE_COMP_N_LABEL(2, k.g.n.edge_r);
	UPDATE_COMP_N_LABEL(3, k.g.n.edge_l);
#if NEIGHBORHOOD == 8
	UPDATE_COMP_N_LABEL(4,k.g.n.edge_dr);
	UPDATE_COMP_N_LABEL(5,k.g.n.edge_dl);
	UPDATE_COMP_N_LABEL(6,k.g.n.edge_ur);
	UPDATE_COMP_N_LABEL(7,k.g.n.edge_ul);
#endif
	k.g.n.comp_n[thread_id] = comp_n;
}

//! Performs one step of the final labeling
/*!
 * This kernel updates neighbors that are connected to labeled nodes, spreading the labels.
 *
 */
__global__ void SpreadLabels(KernelWrapper k, int * label, int * alive) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_id = x + y * k.g.width_ex;
	int label_id = x + y * k.g.width;
	int local_idx = (threadIdx.y + 1) * (THREADS_X + 2) + threadIdx.x + 1;

	int comp_n = k.g.n.comp_n[thread_id];

	__shared__
	int local_label[SHARED_MEMORY_SIZE];

	if (x < k.g.width && y < k.g.height) {

		local_label[local_idx] = label[label_id];

		threadIdx.x == (THREADS_X - 1) && x < k.g.width - 1 ?
				local_label[local_idx + 1] = label[label_id + 1] : 0;
		threadIdx.x == 0 && x > 0 ?
				local_label[local_idx - 1] = label[label_id - 1] : 0;
		threadIdx.y == (THREADS_Y - 1) && y < k.g.height - 1 ?
				local_label[local_idx + (THREADS_X + 2)] = label[label_id
						+ k.g.width] :
				0;
		threadIdx.y == 0 && y > 0 ? local_label[local_idx - (THREADS_X + 2)] =
											label[label_id - k.g.width] :
									0;

#if NEIGHBORHOOD == 8
		threadIdx.x == 0 && threadIdx.y == 0 &&
		x > 0 && y > 0 ? local_label[local_idx - (THREADS_X + 3)] = label[label_id - 1 - k.g.width] : 0;
		threadIdx.x == (THREADS_X - 1) && threadIdx.y == 0 &&
		x < k.g.width - 1 && y > 0 ? local_label[local_idx - (THREADS_X + 1)] = label[label_id + 1 - k.g.width] : 0;
		threadIdx.x == 0 && threadIdx.y == (THREADS_Y - 1) &&
		x > 0 && y < k.g.height - 1 ? local_label[local_idx + (THREADS_X + 1)] = label[label_id - 1 + k.g.width] : 0;
		threadIdx.x == (THREADS_X - 1) && threadIdx.y == (THREADS_Y - 1) &&
		x < k.g.width - 1 && y < k.g.height - 1 ? local_label[local_idx + (THREADS_X + 3)] = label[label_id + 1 + k.g.width] : 0;
#endif
	}

	__syncthreads();

	int curr_label = local_label[local_idx];
	int orig_label = curr_label;

	int repetitions = GLOBAL_RELABEL_LOOPS_PER_KERNEL_CALL;

	do {

#ifdef SPREAD_ZEROS
		if (curr_label && x > 0 && y > 0 && x < k.g.width - 1 && y < k.g.height - 1) {

			curr_label = ( (comp_n & (1<<0)) || local_label[local_idx+(THREADS_X + 2)]) && curr_label;
			curr_label = ( (comp_n & (1<<1)) || local_label[local_idx-(THREADS_X + 2)]) && curr_label;
			curr_label = ( (comp_n & (1<<2)) || local_label[local_idx+1]) && curr_label;
			curr_label = ( (comp_n & (1<<3)) || local_label[local_idx-1]) && curr_label;
#if NEIGHBORHOOD == 8
			curr_label = ( (comp_n & (1<<4)) || local_label[local_idx+(THREADS_X + 3)]) && curr_label;
			curr_label = ( (comp_n & (1<<5)) || local_label[local_idx+(THREADS_X + 1)]) && curr_label;
			curr_label = ( (comp_n & (1<<6)) || local_label[local_idx-(THREADS_X + 1)]) && curr_label;
			curr_label = ( (comp_n & (1<<7)) || local_label[local_idx-(THREADS_X + 3)]) && curr_label;
#endif
#else
		if (x > 0 && y > 0 && x < k.g.width - 1 && y < k.g.height - 1) {
			curr_label = k.g.n.edge_u[thread_id + k.g.width_ex]
					&& local_label[local_idx + (THREADS_X + 2)] || curr_label;
			curr_label = k.g.n.edge_d[thread_id - k.g.width_ex]
					&& local_label[local_idx - (THREADS_X + 2)] || curr_label;
			curr_label = k.g.n.edge_l[thread_id + 1]
					&& local_label[local_idx + 1] || curr_label;
			curr_label = k.g.n.edge_r[thread_id - 1]
					&& local_label[local_idx - 1] || curr_label;
#if NEIGHBORHOOD == 8
			curr_label = k.g.n.edge_ul[thread_id+k.g.width_ex+1] && local_label[local_idx+(THREADS_X + 3)] || curr_label;
			curr_label = k.g.n.edge_ur[thread_id+k.g.width_ex-1] && local_label[local_idx+(THREADS_X + 1)] || curr_label;
			curr_label = k.g.n.edge_dl[thread_id-k.g.width_ex+1] && local_label[local_idx-(THREADS_X + 1)] || curr_label;
			curr_label = k.g.n.edge_dr[thread_id-k.g.width_ex-1] && local_label[local_idx-(THREADS_X + 3)] || curr_label;
#endif
#endif
			curr_label != orig_label ? label[label_id] = curr_label : 0;
			curr_label != orig_label ? local_label[local_idx] = curr_label : 0;
			curr_label != orig_label ? alive[0] = 1 : 0;
			orig_label = curr_label;
		}
		__syncthreads();
	} while (--repetitions);
}

//! Performs the Push operation
/*!
 * This is one of the classical operations from Push-Relabel algorithms. Push is applied to every node once or several times,
 * depending on the parameter PUSHES_PER_KERNEL.
 */
__global__ void /*__launch_bounds__(THREADS_X*THREADS_Y,6)*/ Push(
		KernelWrapper k, int skip, int * alive) {
	if (k.active[blockIdx.x + blockIdx.y * k.block_x] /*__syncthreads_or(k.g.n.status[thread_id] == 1)*/) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int thread_id = x + y * k.g.width_ex;
		//int comp_h = k.g.n.comp_h[thread_id];

		int local_idx = (threadIdx.y + 1) * (THREADS_X + 2) + threadIdx.x + 1;

		__shared__
		int local_height[SHARED_MEMORY_SIZE];

		local_height[local_idx] = k.g.n.height[thread_id];

		threadIdx.x == (THREADS_X - 1) && x < k.g.width_ex - 1 ?
				local_height[local_idx + 1] = k.g.n.height[thread_id + 1] : 0;
		threadIdx.x == 0 && x > 0 ?
				local_height[local_idx - 1] = k.g.n.height[thread_id - 1] : 0;
		threadIdx.y == (THREADS_Y - 1) && y < k.g.height_ex - 1 ?
				local_height[local_idx + (THREADS_X + 2)] =
						k.g.n.height[thread_id + k.g.width_ex] :
				0;
		threadIdx.y == 0 && y > 0 ?
				local_height[local_idx - (THREADS_X + 2)] =
						k.g.n.height[thread_id - k.g.width_ex] :
				0;

#if NEIGHBORHOOD == 8
		threadIdx.x == 0 && threadIdx.y == 0 &&
		x > 0 && y > 0 ? local_height[local_idx - (THREADS_X + 3)] = k.g.n.height[thread_id - 1 - k.g.width_ex] : 0;
		threadIdx.x == (THREADS_X - 1) && threadIdx.y == 0 &&
		x < k.g.width_ex - 1 && y > 0 ? local_height[local_idx - (THREADS_X + 1)] = k.g.n.height[thread_id + 1 - k.g.width_ex] : 0;
		threadIdx.x == 0 && threadIdx.y == (THREADS_Y - 1) &&
		x > 0 && y < k.g.height_ex - 1 ? local_height[local_idx + (THREADS_X + 1)] = k.g.n.height[thread_id - 1 + k.g.width_ex] : 0;
		threadIdx.x == (THREADS_X - 1) && threadIdx.y == (THREADS_Y - 1) &&
		x < k.g.width_ex - 1 && y < k.g.height_ex - 1 ? local_height[local_idx + (THREADS_X + 3)] = k.g.n.height[thread_id + 1 + k.g.width_ex] : 0;
#endif

		bool did_something;

		//if (k.g.n.status[thread_id]) {
		int excess /*= k.g.n.excess[thread_id]*/;
		int cap;
		int flow;
		for (int i = 0; i < PUSHES_PER_KERNEL; ++i) {
			did_something = false;
			excess = k.g.n.excess[thread_id];

			DO_PUSH(k.g.n.edge_l, k.g.n.edge_r, 1, 0, 0, 0, 0);
			DO_PUSH(k.g.n.edge_r, k.g.n.edge_l, 0, 1, 0, 0, 1);
			DO_PUSH(k.g.n.edge_u, k.g.n.edge_d, 0, 0, 1, 0, 2);
			DO_PUSH(k.g.n.edge_d, k.g.n.edge_u, 0, 0, 0, 1, 3);
#if NEIGHBORHOOD == 8
			DO_PUSH(k.g.n.edge_dr,k.g.n.edge_ul,0,1,0,1,4);
			DO_PUSH(k.g.n.edge_dl,k.g.n.edge_ur,1,0,0,1,5);
			DO_PUSH(k.g.n.edge_ur,k.g.n.edge_dl,0,1,1,0,6);
			DO_PUSH(k.g.n.edge_ul,k.g.n.edge_dr,1,0,1,0,7);
#endif

			__threadfence_block();
			if (__syncthreads_and(!did_something))
				break;
		}
		//}
		/*if (!skip && did_something)
		 alive[0] = 1;*/
	}
}

//! Performs the Push operation using the wave-like DO_PUSH macro.
/*!
 * This is one of the classical operations from Push-Relabel algorithms. Push is applied to every node once or several times,
 * depending on the parameter PUSHES_PER_KERNEL.
 */
__global__ void /*__launch_bounds__(THREADS_X*THREADS_Y,6)*/ WavePush(
		KernelWrapper k, const int skip, int * alive) {
	if (k.active[blockIdx.x + blockIdx.y * k.block_x]) {
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;
		const int thread_id = x + y * k.g.width_ex;

		/*if(skip && !threadIdx.x && !threadIdx.y)
		 atomicAdd(&alive[0],1);*/

		//const int comp_h = k.g.n.comp_h[thread_id];
		const int local_idx = (threadIdx.y + 1) * (THREADS_X + 2) + threadIdx.x
				+ 1;

		__shared__
		int local_height[SHARED_MEMORY_SIZE];

		local_height[local_idx] = k.g.n.height[thread_id];

		threadIdx.x == (THREADS_X - 1) && x < k.g.width_ex - 1 ?
				local_height[local_idx + 1] = k.g.n.height[thread_id + 1] : 0;
		threadIdx.x == 0 && x > 0 ?
				local_height[local_idx - 1] = k.g.n.height[thread_id - 1] : 0;
		threadIdx.y == (THREADS_Y - 1) && y < k.g.height_ex - 1 ?
				local_height[local_idx + (THREADS_X + 2)] =
						k.g.n.height[thread_id + k.g.width_ex] :
				0;
		threadIdx.y == 0 && y > 0 ?
				local_height[local_idx - (THREADS_X + 2)] =
						k.g.n.height[thread_id - k.g.width_ex] :
				0;

#if NEIGHBORHOOD == 8
		threadIdx.x == 0 && threadIdx.y == 0 &&
		x > 0 && y > 0 ? local_height[local_idx - (THREADS_X + 3)] = k.g.n.height[thread_id - 1 - k.g.width_ex] : 0;
		threadIdx.x == (THREADS_X - 1) && threadIdx.y == 0 &&
		x < k.g.width_ex - 1 && y > 0 ? local_height[local_idx - (THREADS_X + 1)] = k.g.n.height[thread_id + 1 - k.g.width_ex] : 0;
		threadIdx.x == 0 && threadIdx.y == (THREADS_Y - 1) &&
		x > 0 && y < k.g.height_ex - 1 ? local_height[local_idx + (THREADS_X + 1)] = k.g.n.height[thread_id - 1 + k.g.width_ex] : 0;
		threadIdx.x == (THREADS_X - 1) && threadIdx.y == (THREADS_Y - 1) &&
		x < k.g.width_ex - 1 && y < k.g.height_ex - 1 ? local_height[local_idx + (THREADS_X + 3)] = k.g.n.height[thread_id + 1 + k.g.width_ex] : 0;
#endif

		__shared__
		int local_excess[SHARED_MEMORY_SIZE];

		local_excess[local_idx] = k.g.n.excess[thread_id];

		threadIdx.x == (THREADS_X - 1) && x < k.g.width_ex - 1 ?
				local_excess[local_idx + 1] = 0 : 0;
		threadIdx.x == 0 && x > 0 ? local_excess[local_idx - 1] = 0 : 0;
		threadIdx.y == (THREADS_Y - 1) && y < k.g.height_ex - 1 ?
				local_excess[local_idx + (THREADS_X + 2)] = 0 : 0;
		threadIdx.y == 0 && y > 0 ? local_excess[local_idx - (THREADS_X + 2)] =
											0 :
									0;

#if NEIGHBORHOOD == 8
		threadIdx.x == 0 && threadIdx.y == 0 && x > 0 && y > 0 ? local_excess[local_idx -(THREADS_X + 3)] = 0 : 0;
		threadIdx.x == (THREADS_X - 1)&& threadIdx.y == 0 && x < k.g.width_ex - 1 && y > 0 ? local_excess[local_idx -(THREADS_X + 1)] = 0 : 0;
		threadIdx.x == 0 && threadIdx.y == (THREADS_Y - 1) && x > 0 && y < k.g.height_ex - 1 ? local_excess[local_idx +(THREADS_X + 1)] = 0 : 0;
		threadIdx.x == (THREADS_X - 1)&& threadIdx.y == (THREADS_Y - 1) && x < k.g.width_ex - 1 && y < k.g.height_ex - 1 ? local_excess[local_idx +(THREADS_X + 3)] = 0 : 0;
#endif

		__syncthreads();

		bool did_something;

		const int original_excess = local_excess[local_idx];

		__syncthreads(); // only for determinism

		//if (k.g.n.status[thread_id]) {
		int excess;
		int cap;
		int flow;
		for (int i = 0; i < PUSHES_PER_KERNEL; ++i) {
			did_something = false;

			DO_PUSH_WAVE2(k.g.n.edge_l, k.g.n.edge_r, 1, 0, 0, 0, 0);
			DO_PUSH_WAVE2(k.g.n.edge_r, k.g.n.edge_l, 0, 1, 0, 0, 1);
			DO_PUSH_WAVE2(k.g.n.edge_u, k.g.n.edge_d, 0, 0, 1, 0, 2);
			DO_PUSH_WAVE2(k.g.n.edge_d, k.g.n.edge_u, 0, 0, 0, 1, 3);
#if NEIGHBORHOOD == 8
			DO_PUSH_WAVE2(k.g.n.edge_dr,k.g.n.edge_ul,0,1,0,1,4);
			DO_PUSH_WAVE2(k.g.n.edge_dl,k.g.n.edge_ur,1,0,0,1,5);
			DO_PUSH_WAVE2(k.g.n.edge_ur,k.g.n.edge_dl,0,1,1,0,6);
			DO_PUSH_WAVE2(k.g.n.edge_ul,k.g.n.edge_dr,1,0,1,0,7);
#endif
			if (__syncthreads_and(!did_something))
				break;

			//excess = k.g.n.excess[thread_id];
		}
		excess = local_excess[local_idx];

		/*if(threadIdx.x > 0 && threadIdx.x < (THREADS_X - 1) && threadIdx.y > 0 && threadIdx.y < (THREADS_Y - 1))
		 excess - original_excess ? k.g.n.excess[thread_id] = excess : 0;
		 else*/
		excess - original_excess ?
				atomicAdd(&k.g.n.excess[thread_id], excess - original_excess) :
				0;

		threadIdx.x == 0 && x > 0 && local_excess[local_idx - 1] ?
				atomicAdd(&k.g.n.excess[thread_id - 1],
						local_excess[local_idx - 1]) :
				0;
		threadIdx.y == 0 && y > 0 && local_excess[local_idx - (THREADS_X + 2)] ?
				atomicAdd(&k.g.n.excess[thread_id - k.g.width_ex],
						local_excess[local_idx - (THREADS_X + 2)]) :
				0;
		threadIdx.x == (THREADS_X - 1) && x < k.g.width_ex - 1
				&& local_excess[local_idx + 1] ?
				atomicAdd(&k.g.n.excess[thread_id + 1],
						local_excess[local_idx + 1]) :
				0;
		threadIdx.y == (THREADS_Y - 1) && y < k.g.height_ex - 1
				&& local_excess[local_idx + (THREADS_X + 2)] ?
				atomicAdd(&k.g.n.excess[thread_id + k.g.width_ex],
						local_excess[local_idx + (THREADS_X + 2)]) :
				0;
#if NEIGHBORHOOD == 8
		threadIdx.x == 0 && x > 0 && threadIdx.y == 0 && y > 0
		&& local_excess[local_idx - (THREADS_X + 3)] ?
		atomicAdd(&k.g.n.excess[thread_id - (k.g.width_ex + 1)],
				local_excess[local_idx - (THREADS_X + 3)]) :
		0;
		threadIdx.x == (THREADS_X - 1) && x < k.g.width_ex - 1 && threadIdx.y == 0
		&& y > 0 && local_excess[local_idx - (THREADS_X + 1)] ?
		atomicAdd(&k.g.n.excess[thread_id - (k.g.width_ex - 1)],
				local_excess[local_idx - (THREADS_X + 1)]) :
		0;
		threadIdx.x == 0 && x > 0 && threadIdx.y == (THREADS_Y - 1)
		&& y < k.g.height_ex - 1 && local_excess[local_idx + (THREADS_X + 1)] ?
		atomicAdd(&k.g.n.excess[thread_id + (k.g.width_ex - 1)],
				local_excess[local_idx + (THREADS_X + 1)]) :
		0;
		threadIdx.x == (THREADS_X - 1) && x < k.g.width_ex - 1
		&& threadIdx.y == (THREADS_Y - 1) && y < k.g.height_ex - 1
		&& local_excess[local_idx + (THREADS_X + 3)] ?
		atomicAdd(&k.g.n.excess[thread_id + (k.g.width_ex + 1)],
				local_excess[local_idx + (THREADS_X + 3)]) :
		0;
#endif
		//}
		/*if (!skip && did_something)
		 alive[0] = 1;*/
	}
}

/*! \def ADJUST_HEIGHT(diff,edge)
 * \brief Updates the current height considering the neighbor in one direction.
 */
#define ADJUST_HEIGHT(diff,edge) (height > local_height[local_idx + (diff)] && (edge) > 0) ? height = local_height[local_idx + (diff)] : 0

/*! \def UPDATE_COMP_H(i,diff)
 * \brief Updates the compressed height for one direction. Compressed heights will be used in Push.
 *
 *  Updates by checking possibility to push to the neighbor in a certain direction. Only considers height compatibility.
 */
#define UPDATE_COMP_H(i,diff) comp_h |= (1 << (i)) * (local_height[local_idx] == local_height[local_idx+(diff)] + 1)

//! Performs the Relabel operation.
/*!
 * This is one of the classical operations from Push-Relabel algorithms. Relabel is applied to every node once or several times,
 * depending on the parameter RELABELS_PER_KERNEL.
 */
__global__ void /*__launch_bounds__(THREADS_X*THREADS_Y,6)*/ Relabel(
		KernelWrapper k, const int skip) {
	if (!skip
			|| k.active[blockIdx.x + blockIdx.y * k.block_x] /*__syncthreads_or(status == 1)*/) {
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;
		const int thread_id = x + y * k.g.width_ex;

		int status = k.g.n.status[thread_id];

		__shared__
		int local_height[SHARED_MEMORY_SIZE];

		const int local_idx = (threadIdx.y + 1) * (THREADS_X + 2) + threadIdx.x
				+ 1;

		local_height[local_idx] = k.g.n.height[thread_id];

		threadIdx.x == (THREADS_X - 1) && x < k.g.width_ex - 1 ?
				local_height[local_idx + 1] = k.g.n.height[thread_id + 1] : 0;
		threadIdx.x == 0 && x > 0 ?
				local_height[local_idx - 1] = k.g.n.height[thread_id - 1] : 0;
		threadIdx.y == (THREADS_Y - 1) && y < k.g.height_ex - 1 ?
				local_height[local_idx + (THREADS_X + 2)] =
						k.g.n.height[thread_id + k.g.width_ex] :
				0;
		threadIdx.y == 0 && y > 0 ?
				local_height[local_idx - (THREADS_X + 2)] =
						k.g.n.height[thread_id - k.g.width_ex] :
				0;

#if NEIGHBORHOOD == 8
		threadIdx.x == 0 && threadIdx.y == 0 &&
		x > 0 && y > 0 ? local_height[local_idx - (THREADS_X + 3)] = k.g.n.height[thread_id - 1 - k.g.width_ex] : 0;
		threadIdx.x == (THREADS_X - 1) && threadIdx.y == 0 &&
		x < k.g.width_ex - 1 && y > 0 ? local_height[local_idx - (THREADS_X + 1)] = k.g.n.height[thread_id + 1 - k.g.width_ex] : 0;
		threadIdx.x == 0 && threadIdx.y == (THREADS_Y - 1) &&
		x > 0 && y < k.g.height_ex - 1 ? local_height[local_idx + (THREADS_X + 1)] = k.g.n.height[thread_id - 1 + k.g.width_ex] : 0;
		threadIdx.x == (THREADS_X - 1) && threadIdx.y == (THREADS_Y - 1) &&
		x < k.g.width_ex - 1 && y < k.g.height_ex - 1 ? local_height[local_idx + (THREADS_X + 3)] = k.g.n.height[thread_id + 1 + k.g.width_ex] : 0;
#endif

		__syncthreads();

		//bool did_something = false;

		const int excess = k.g.n.excess[thread_id];

		int edge_l, edge_r, edge_u, edge_d;
#if NEIGHBORHOOD == 8
		int edge_ul, edge_ur, edge_dl, edge_dr;
#endif
		if (status > 0) {
			edge_l = k.g.n.edge_l[thread_id];
			edge_r = k.g.n.edge_r[thread_id];
			edge_u = k.g.n.edge_u[thread_id];
			edge_d = k.g.n.edge_d[thread_id];
#if NEIGHBORHOOD == 8
			edge_ul = k.g.n.edge_ul[thread_id];
			edge_ur = k.g.n.edge_ur[thread_id];
			edge_dl = k.g.n.edge_dl[thread_id];
			edge_dr = k.g.n.edge_dr[thread_id];
#endif
		}

		int height = -1;

		for (int i = 0; i < RELABELS_PER_KERNEL; ++i) {
			//bool changed = false;
			if (excess >= 0 && status > 0) {
				height = k.g.size;
				ADJUST_HEIGHT( -1, edge_l);
				ADJUST_HEIGHT( 1, edge_r);
				ADJUST_HEIGHT(-(THREADS_X + 2), edge_u);
				ADJUST_HEIGHT( (THREADS_X + 2), edge_d);
#if NEIGHBORHOOD == 8
				ADJUST_HEIGHT(-(THREADS_X + 3),edge_ul);
				ADJUST_HEIGHT(-(THREADS_X + 1),edge_ur);
				ADJUST_HEIGHT( (THREADS_X + 1),edge_dl);
				ADJUST_HEIGHT( (THREADS_X + 3),edge_dr);
#endif
				//height != k.g.size_ex ? printf("Changed from %d to %d\n",k.g.n.height[thread_id],height): 0;
				//status = height != k.g.size;

				/*if(local_height[local_idx] != height + 1) {
				 changed |= true;
				 }*/

				local_height[local_idx] = height + 1;
			}
			/*if(__syncthreads_and(!changed))
			 break;*/
			__syncthreads();
			//__syncthreads_and(!changed);
		}
		if (height != -1)
			k.g.n.height[thread_id] = height + 1;

		/*__syncthreads();
		 int comp_h = 0;
		 if (x > 0 && y > 0 && x < k.g.width - 1 && y < k.g.height - 1) {
			 UPDATE_COMP_H(0, -1);
			 UPDATE_COMP_H(1, 1);
			 UPDATE_COMP_H(2, -(THREADS_X + 2));
			 UPDATE_COMP_H(3, (THREADS_X + 2));
			 #if NEIGHBORHOOD == 8
			 UPDATE_COMP_H(7,-(THREADS_X + 3));
			 UPDATE_COMP_H(6,-(THREADS_X + 1));
			 UPDATE_COMP_H(5, (THREADS_X + 1));
			 UPDATE_COMP_H(4, (THREADS_X + 3));
			 #endif
		 }
		 k.g.n.comp_h[thread_id] = comp_h;*/

		if (status) {
			if (height == k.g.size)
				k.g.n.status[thread_id] = 0;
			else
				k.g.n.status[thread_id] = 2 - (excess > 0);
		}
		//k.g.n.status[thread_id] = status;
		/*if(did_something)
		 alive[0] = 1;*/
	}
}

//! Updates the activity status of each block by reading the status of each thread.
/*!
 */
__global__ void UpdateActivity(int * status, int * active, int block_x,
		int width_ex, int * alive) {
	int block_id = blockIdx.x + blockIdx.y * block_x;
	active[block_id] = 0;

	__syncthreads();

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_id = x + y * width_ex;

	status[thread_id] == 1 ? active[block_id] = 1 : 0;

	status[thread_id] == 1 ? alive[0] = 1 : 0;
}

/*! \def UPDATE_COMP_N(i,diff)
 * \brief Updates the compressed neighborhood.
 *
 *  Updates by checking possibility to push to the neighbor in a certain direction. Only considers edge capacity.
 */
#define UPDATE_COMP_N(i,edge) comp_n |= (1<<(i)) * (edge[thread_id] > 0)

//! Initializes global relabeling.
/*!
 * This kernel resets heights and updates compressed neighborhoods to prepare for global relabeling.
 *
 */
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
	UPDATE_COMP_N(0, k.g.n.edge_l);
	UPDATE_COMP_N(1, k.g.n.edge_r);
	UPDATE_COMP_N(2, k.g.n.edge_u);
	UPDATE_COMP_N(3, k.g.n.edge_d);
#if NEIGHBORHOOD == 8
	UPDATE_COMP_N(4,k.g.n.edge_ul);
	UPDATE_COMP_N(5,k.g.n.edge_ur);
	UPDATE_COMP_N(6,k.g.n.edge_dl);
	UPDATE_COMP_N(7,k.g.n.edge_dr);
#endif
	/* different because of >= instead of > */
	/* and accounting for activity instead of only no-sinkness */
	comp_n |= (1 << 8) * no_sink;
	k.g.n.comp_n[thread_id] = comp_n;
}

/*! \def COMP_ADJUST_HEIGHT(diff,i)
 * \brief Updates the current height considering the neighbor in one direction, and using the compressed neighborhood.
 */
#define COMP_ADJUST_HEIGHT(diff,i) (height > local_height[local_idx + (diff)] && ((1<<(i)) & comp_n)) ? height = local_height[local_idx + (diff)] : 0

//! Performs one step of a Global Relabel, and sets alive to true if the kernel should be run again.
/*!
 * Global relabeling sets the heights of the nodes to the distance between them and the closest sink-connected node. This
 * tends to help make flow go in the right direction, decreasing the number of necessary iterations for the algorithm.
 *
 */
__global__ void GlobalRelabel(KernelWrapper k, int * alive) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_id = x + y * k.g.width_ex;

	int comp_n = k.g.n.comp_n[thread_id];
	if (__syncthreads_or(comp_n & (1 << 8))) {

		__shared__
		int local_height[SHARED_MEMORY_SIZE];

		int local_idx = (threadIdx.y + 1) * (THREADS_X + 2) + threadIdx.x + 1;

		local_height[local_idx] = k.g.n.height[thread_id];

		/*int outer_repetitions = 1;

		 do {*/
		threadIdx.x == (THREADS_X - 1) && x < k.g.width_ex - 1 ?
				local_height[local_idx + 1] = k.g.n.height[thread_id + 1] : 0;
		threadIdx.x == 0 && x > 0 ?
				local_height[local_idx - 1] = k.g.n.height[thread_id - 1] : 0;
		threadIdx.y == (THREADS_Y - 1) && y < k.g.height_ex - 1 ?
				local_height[local_idx + (THREADS_X + 2)] =
						k.g.n.height[thread_id + k.g.width_ex] :
				0;
		threadIdx.y == 0 && y > 0 ?
				local_height[local_idx - (THREADS_X + 2)] =
						k.g.n.height[thread_id - k.g.width_ex] :
				0;

#if NEIGHBORHOOD == 8
		threadIdx.x == 0 && threadIdx.y == 0 &&
		x > 0 && y > 0 ? local_height[local_idx - (THREADS_X + 3)] = k.g.n.height[thread_id - 1 - k.g.width_ex] : 0;
		threadIdx.x == (THREADS_X - 1) && threadIdx.y == 0 &&
		x < k.g.width_ex - 1 && y > 0 ? local_height[local_idx - (THREADS_X + 1)] = k.g.n.height[thread_id + 1 - k.g.width_ex] : 0;
		threadIdx.x == 0 && threadIdx.y == (THREADS_Y - 1) &&
		x > 0 && y < k.g.height_ex - 1 ? local_height[local_idx + (THREADS_X + 1)] = k.g.n.height[thread_id - 1 + k.g.width_ex] : 0;
		threadIdx.x == (THREADS_X - 1) && threadIdx.y == (THREADS_Y - 1) &&
		x < k.g.width_ex - 1 && y < k.g.height_ex - 1 ? local_height[local_idx + (THREADS_X + 3)] = k.g.n.height[thread_id + 1 + k.g.width_ex] : 0;
#endif

		__syncthreads();

		bool changed = false;

		int repetitions = GLOBAL_RELABEL_LOOPS_PER_KERNEL_CALL;
		do {
			int height = local_height[local_idx] - 1;
			if (((1 << 8) & comp_n) && x > 0 && y > 0 && x < k.g.width - 1
					&& y < k.g.height - 1) {
				height = k.g.size;

				COMP_ADJUST_HEIGHT( -1, 0);
				COMP_ADJUST_HEIGHT( 1, 1);
				COMP_ADJUST_HEIGHT(-(THREADS_X + 2), 2);
				COMP_ADJUST_HEIGHT( (THREADS_X + 2), 3);
#if NEIGHBORHOOD == 8
				COMP_ADJUST_HEIGHT(-(THREADS_X + 3), 4);
				COMP_ADJUST_HEIGHT(-(THREADS_X + 1), 5);
				COMP_ADJUST_HEIGHT( (THREADS_X + 1), 6);
				COMP_ADJUST_HEIGHT( (THREADS_X + 3), 7);
#endif
			}
			__syncthreads();
			changed |= (local_height[local_idx] != height + 1);
			local_height[local_idx] = height + 1;
			__syncthreads();
		} while (--repetitions);
		//height != k.g.size_ex ? printf("Changed from %d to %d\n",k.g.n.height[thread_id],height): 0;
		changed ? k.g.n.comp_n[thread_id] = comp_n ^ (1 << 8) : 0;
		changed ? k.g.n.height[thread_id] = local_height[local_idx] : 0;
		//k.g.n.height[thread_id] = height + 1;

		//__syncthreads();

		changed ? alive[0] = 1 : 0;
		//} while(--outer_repetitions);
	}
}

//! Calculates the total energy level in the current state of the graph.
/*!
 * This function parallelly sums up the positive values of \a source, and the
 * result shall be in the 1st position of \a output.
 *
 */
__global__ void EnergyLevel(KernelWrapper k, int stride, int odd,
		int * source, int * output, bool limited) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_id = x + y * k.g.width_ex;
	int source_id = thread_id;
	if(limited)
		source_id = x + y * k.g.width;

	if ((stride == k.g.size_ex && !limited) || (stride == k.g.size && limited)) {
		if(!limited || (x >= 0 && y >= 0 && x <= k.g.width - 1 && y <= k.g.height - 1))
			output[thread_id] = source[source_id] > 0 ? source[source_id] : 0;
			//output[thread_id] = k.g.n.height[thread_id];
	} else if (thread_id < stride && (!odd || thread_id != stride - 1))
		output[thread_id] += output[thread_id + stride];
}

#endif /* GRAPHCUTKERNELS_CU_ */
