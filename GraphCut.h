
/*!
 * \file GraphCut.h
 *
 * \author bruno
 * \date Jun 8, 2012
 *
 * Header file to hold the definitions of data structures that are necessary for the Graph Cut algorithm.
 */

#ifndef GRAPHCUT_H_
#define GRAPHCUT_H_

#include <cuda.h>
#include <cutil.h>
#include <cutil_inline.h>

/*! \def NEIGHBORHOOD
 * \brief Defines the adjancency of the graph.
 *
 * A defined value that should hold either 4 or 8, corresponding to 4 or 8 adjacency, respectively.
 */
#define NEIGHBORHOOD 8

/*! \def DEBUG_MODE
 * \brief Turns debug and verbose modes on.
 *
 * If this is defined, intermediate values are copied to the host and printed out during each iteration
 * of the Graph Cut algorithm.
 */
//#define DEBUG_MODE

/*! \def SPREAD_ZEROS
 * \brief Makes undecided pixels get the standard value of true or 1.
 *
 * If this is defined, labeling will spread zeros instead of ones, and this will cause disconnected pixels to
 * remain with the value true or 1.
 */
#define SPREAD_ZEROS

/*! \def THREADS_X
 * \brief Corresponds to the width of a block.
 * Corresponds to the width of a block.
 */
#define THREADS_X 32
/*! \def THREADS_Y
 * \brief Corresponds to the height of a block.
 * Corresponds to the height of a block.
 */
#define THREADS_Y 8
/*! \def THREAD_COUNT
 * \brief Number of threads per block.
 * Number of threads per block.
 */
#define THREAD_COUNT ((THREADS_X)*(THREADS_Y))

/*! \struct NodeWrapper
 * \brief A struct to hold the per node information.
 *
 * This struct contains data that exists for each pixel, and is the main representation of the graph within the algorithm.
 */
struct NodeWrapper {
	int * edge_l; //!< Left edge.
	int * edge_r; //!< Right edge.
	int * edge_u; //!< Up edge.
	int * edge_d; //!< Down edge.
#if NEIGHBORHOOD == 8
	int * edge_ul; //!< Up-left edge.
	int * edge_dr; //!< Down-right edge.
	int * edge_ur; //!< Up-right edge.
	int * edge_dl; //!< Down-left edge.
#endif
	int * height; //!< Current height of a node
	int * excess; //!< Current flow in the node. Can be negative, indicating an existing connection with the sink node.
	int * status; //!< Activity status of a node (whether it can push). Used to determine active blocks.
	int * comp_h; //!< Compressed height information relating neighbors and the node itself. Used in Push.
	int * comp_n; //!< Compressed neighborhood information. Used in Global Relabel.
};

/*! \struct GraphWrapper
 * \brief A struct to hold the graph information.
 *
 * This struct contains a NodeWrapper and other general information about the graph.
 */
struct GraphWrapper {
	int width; //!< Graph original width.
	int height; //!< Graph original height.
	int size; //!< Graph original size.
	int width_ex; //!< Graph width adapted to block size.
	int height_ex; //!< Graph height adapted to block size.
	int size_ex; //!< Graph size adapted to block size.
	NodeWrapper n; //!< The node wrapper.
};

/*! \struct KernelWrapper
 * \brief A struct to hold information necessary to the kernels.
 *
 * This struct contains a GraphWrapper and block information.
 */
struct KernelWrapper {
	int block_x; //!< The number of blocks in the horizontal direction.
	int * active; //!< Holds for every block whether or not it is active.
	GraphWrapper g; //!< The graph wrapper
};

/*! \struct GlobalWrapper
 * \brief A struct to hold information necessary to any step of Graph Cut.
 *
 * This struct contains a KernelWrapper, block information and a copy of original data terms and edge values.
 */
struct GlobalWrapper {
	int block_y; //!< Number of blocks in the vertical direction.
	int block_count; //!< Number of blocks.
	int * data_positive; //!< Original positive data term.
	int * data_negative; //!< Original negative data term.
	int * up; //!< Original up edge.
	int * down; //!< Original down edge.
	int * left; //!< Original left edge.
	int * right; //!< Original right edge.
#if NEIGHBORHOOD == 8
	int * upleft; //!< Original up-left edge.
	int * upright; //!< Original up-right edge.
	int * downleft; //!< Original down-left edge.
	int * downright; //!< Original down-right edge.
#endif
	int penalty;  //!< Standard edge value when no per edge value has been given.
	bool varying_edges;  //!< Whether edge values have been given.
	KernelWrapper k; //!< The kernel wrapper.
};


#endif /* GRAPHCUT_H_ */
