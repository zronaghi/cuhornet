/*
Please cite:
* J. Fox, O. Green, K. Gabert, X. An, D. Bader, “Fast and Adaptive List Intersections on the GPU”, 
IEEE High Performance Extreme Computing Conference (HPEC), 
Waltham, Massachusetts, 2018
* O. Green, J. Fox, A. Tripathy, A. Watkins, K. Gabert, E. Kim, X. An, K. Aatish, D. Bader, 
“Logarithmic Radix Binning and Vectorized Triangle Counting”, 
IEEE High Performance Extreme Computing Conference (HPEC), 
Waltham, Massachusetts, 2018
* O. Green, P. Yalamanchili ,L.M. Munguia, “Fast Triangle Counting on GPU”, 
Irregular Applications: Architectures and Algorithms (IA3), 
New Orleans, Louisiana, 2014 
*/

#include <cuda.h>
#include <cuda_runtime.h>

#include "Static/SpGEMM/spgemm.cuh"
#include "Static/SpGEMM/spgemm-Operators.cuh"
#include "StandardAPI.hpp"
#include <Operator++.cuh>

namespace hornets_nest {

SpGEMM::SpGEMM(HornetGraph* hornetA_, HornetGraph* hornetB_, HornetGraph* hornetC_, 
               int concurrentIntersections_, float workFactor_, bool sanityCheck_) {

    hornetA = hornetA_; 
    hornetB = hornetB_; 
    hornetC = hornetC_;
    concurrentIntersections = concurrentIntersections_;
    workFactor = workFactor_;
    sanityCheck = sanityCheck_;

    gpu::allocate(vertex_pairs, hornetC->nV());

    reset();

}
SpGEMM::~SpGEMM(){
    release();
}

SpGEMM::SpGEMM(HornetGraphPtr* hornetA_, HornetGraphPtr* hornetB_, HornetGraphPtr* hornetC_, 
               int numGPUs_, int concurrentIntersections_, float workFactor_, bool sanityCheck_) {

    // hornetA = hornetA_; 
    // hornetB = hornetB_; 
    // hornetC = hornetC_;
    concurrentIntersections = concurrentIntersections_;
    workFactor = workFactor_;
    sanityCheck = sanityCheck_;
    numGPUs = numGPUs_;

    printf("Move to run func and duplicate across devices");
    gpu::allocate(vertex_pairs, hornetC->nV());

    reset();

}

/*
__global__ void CreateIndexPair (vid_t rowc,
                                 vid_t colc,
                                 vid2_t* vertex_pairs) { 

vid_t row = blockIdx.y * blockDim.y + threadIdx.y;
vid_t col = blockIdx.x * blockDim.x + threadIdx.x;
if (row >= rowc || col >= colc) return;

vertex_pairs [row*colc+col].x = row;
vertex_pairs [row*colc+col].y = col;
}
*/

template <typename vid_t>
struct queue_info_spgemm {
    unsigned long long *d_queue_sizes;
    vid_t *d_edge_queue; // both balanced and imbalanced cases
    unsigned long long *d_queue_pos;
};

template <typename vid_t, typename hornetDevice>
__global__ void bin_vertex_pair_spgemm (hornetDevice hornetDeviceA,
                                        hornetDevice hornetDeviceB,
                                        queue_info_spgemm<vid_t> d_queue_info,
                                        bool countOnly, float workFactor,
                                        int startRow) {

    int bin_index;
    vid_t row = blockIdx.x + startRow;
    // const auto MAX_ADJ_UNIONS_BINS_DIV_2  = MAX_ADJ_UNIONS_BINS/2;
    //vid_t col = blockIdx.x * blockDim.x + threadIdx.x;
        // if (row >= hornetDeviceA.nV() || col >= hornetDeviceB.nV()) return;
    if (row >= hornetDeviceA.nV()) return;
    //     // Choose the bin to place this edge into
    degree_t row_len = hornetDeviceA.vertex(row).degree();

    if(row_len==0)
        return;

    __shared__ int32_t localBins[MAX_ADJ_UNIONS_BINS];
    __shared__ int32_t localPos[MAX_ADJ_UNIONS_BINS];
    int tid =  threadIdx.x; //+ blockDim.x*threadIdx.y;
    int stride = blockDim.x;
    for (int tid2= tid; tid2<MAX_ADJ_UNIONS_BINS; tid2+=stride){
      localBins[tid2]=0;
      localPos[tid2] =0;
    }
    __syncthreads();

    for (vid_t col = tid; col < hornetDeviceB.nV(); col += stride){

        degree_t col_len = hornetDeviceB.vertex(col).degree();
        if(col_len==0)
            continue;

        degree_t u_len = (row_len <= col_len) ? row_len : col_len;
        degree_t v_len = (row_len <= col_len) ? col_len : row_len;

        // degree_t u_len = row_len;
        // degree_t v_len = col_len;
        unsigned int log_v = std::min(32-__clz(v_len), 31);
        unsigned int log_u = std::min(32-__clz(u_len), 31);
        if(log_u==0 || log_v==0)
            continue;
        int binary_work_est = u_len*log_v;
        int intersect_work_est = u_len + v_len + log_u;
        int METHOD = ((workFactor*(float)intersect_work_est >= (float)binary_work_est));
        //// METHOD == 0 is binary search, METHOD == 1 intersection 
        if ( METHOD == 0 && u_len <= 1) {
            bin_index = 0;
        } else if (METHOD == 0) {
            bin_index = (log_v*BINS_1D_DIM+log_u);
        } else {
            // bin_index = (MAX_ADJ_UNIONS_BINS/2)+(log_u*BINS_1D_DIM+log_v); 
            bin_index = (MAX_ADJ_UNIONS_BINS/2)+(log_v*BINS_1D_DIM+log_u); 
        }

        atomicAdd(localBins + bin_index, 1ULL);
    }

    __syncthreads();

    for (int tid2= tid; tid2<MAX_ADJ_UNIONS_BINS; tid2+=stride){
        if(localBins [tid2] == 0)
            continue;
        if(countOnly){
            atomicAdd(&(d_queue_info.d_queue_sizes[tid2]), localBins [tid2]);
        }
        else {
            localPos[tid2]=atomicAdd(&(d_queue_info.d_queue_pos[tid2]), localBins[tid2]);
        }
    }
    __syncthreads();

    if(countOnly){
        return;
    }
    else {
        for (vid_t col = tid; col < hornetDeviceB.nV(); col += stride){

            degree_t col_len = hornetDeviceB.vertex(col).degree();
            if(col_len==0)
                continue;

            degree_t u_len = (row_len <= col_len) ? row_len : col_len;
            degree_t v_len = (row_len <= col_len) ? col_len : row_len;

            // degree_t u_len = row_len;
            // degree_t v_len = col_len;
            unsigned int log_v = std::min(32-__clz(v_len), 31);
            unsigned int log_u = std::min(32-__clz(u_len), 31);
            // if(log_u==0 || log_v==0)
            //     continue;


            int binary_work_est = u_len*log_v;
            int intersect_work_est = u_len + v_len + log_u;
            // int METHOD = ((WORK_FACTOR*intersect_work_est >= binary_work_est));
            int METHOD = ((workFactor*(float)intersect_work_est >= (float)binary_work_est));

            //// METHOD == 0 is binary search, METHOD == 1 intersection 
            if ( METHOD == 0 && u_len <= 1) {
                bin_index = 0;
            } else if (METHOD == 0) {
                bin_index = (log_v*BINS_1D_DIM+log_u);
            } else {
                // bin_index = (MAX_ADJ_UNIONS_BINS/2)+(log_u*BINS_1D_DIM+log_v); 
                bin_index = (MAX_ADJ_UNIONS_BINS/2)+(log_v*BINS_1D_DIM+log_u); 
            }
            
            unsigned long long id = atomicAdd(localPos+bin_index, 1ULL);
            d_queue_info.d_edge_queue[id*2] = row;
            d_queue_info.d_edge_queue[id*2+1] = col;
        }
    }  	
}

template<typename hornetDevice, typename T, typename Operator>
__global__ void forAllEdgesAdjUnionImbalancedKernelSpGEMM(hornetDevice hornetDeviceA, 
													   hornetDevice hornetDeviceB, 
													   T* __restrict__ array, 
													   unsigned long long start, 
													   unsigned long long end, 
													   unsigned long long threads_per_union, 
													   int flag, Operator op, 
                                                       int startRow) {

	// if (blockIdx.x==0)
	// printf("%d %d\n,",threadIdx.x,array[threadIdx.x]);
    // using namespace adj_union;
    // using vid_t = typename hornetDevice::VertexType;
    auto id = blockIdx.x * blockDim.x + threadIdx.x;
    auto queue_id = id / threads_per_union;
    auto block_union_offset = blockIdx.x % ((threads_per_union+blockDim.x-1) / blockDim.x); // > 1 if threads_per_union > block size
    auto thread_union_id = ((block_union_offset*blockDim.x)+threadIdx.x) % threads_per_union;
    auto stride = blockDim.x * gridDim.x;
    auto queue_stride = stride / threads_per_union;
    for (auto i = start+queue_id; i < end; i += queue_stride) {
        auto row_vtx = hornetDeviceA.vertex(array[2*i]);
        auto col_vtx = hornetDeviceB.vertex(array[2*i+1]);
        int rowLen = row_vtx.degree();
        int colLen = col_vtx.degree();
        vid_t row = row_vtx.id();
        vid_t col = col_vtx.id();

		// printf("i=%d, array[i] = %d \n", i, array[i]); 
        // determine u,v where |adj(u)| <= |adj(v)|
        bool sourceSmaller = rowLen < colLen;
        vid_t u = sourceSmaller ? row : col;
        vid_t v = sourceSmaller ? col : row;
        degree_t u_len = sourceSmaller ? rowLen : colLen;
        degree_t v_len = sourceSmaller ? colLen : rowLen;
        auto u_vtx = sourceSmaller ? row_vtx : col_vtx;
        auto v_vtx = sourceSmaller ? col_vtx : row_vtx;
        wgt0_t *u_weight, *v_weight; 
        // vid_t* u_nodes = hornet.vertex(u).neighbor_ptr();
        // vid_t* v_nodes = hornet.vertex(v).neighbor_ptr();
        vid_t *u_nodes, *v_nodes;
        if(sourceSmaller){
            u_nodes = hornetDeviceA.vertex(u).neighbor_ptr();
            v_nodes = hornetDeviceB.vertex(v).neighbor_ptr();
            u_weight = hornetDeviceA.vertex(u).template field_ptr <0> ();
            v_weight = hornetDeviceB.vertex(v).template field_ptr <0> ();
        }else{
            u_nodes = hornetDeviceB.vertex(u).neighbor_ptr();
            v_nodes = hornetDeviceA.vertex(v).neighbor_ptr();
            u_weight = hornetDeviceB.vertex(u).template field_ptr <0> ();
            v_weight = hornetDeviceA.vertex(v).template field_ptr <0> ();
        }

        int ui_begin, vi_begin, ui_end, vi_end;
        vi_begin = 0;
        vi_end = v_len-1;
        auto work_per_thread = u_len / threads_per_union;
        auto remainder_work = u_len % threads_per_union;
        // divide up work evenly among neighbors of u
        ui_begin = thread_union_id*work_per_thread + std::min(thread_union_id, remainder_work);
        ui_end = (thread_union_id+1)*work_per_thread + std::min(thread_union_id+1, remainder_work) - 1;

        int flag2=flag;
        // if (sourceSmaller)
        //     flag2 = flag+2;
        if (row != u)
            flag2 = flag+2;


        if (ui_end < u_len) {
            op(u_vtx, v_vtx, u_nodes+ui_begin, u_nodes+ui_end, u_weight+ui_begin, v_nodes+vi_begin, v_nodes+vi_end, v_weight+vi_begin, flag2, startRow);
        }
    }
}

////ToDo: rewrite the function, binary search
template <typename HornetGraph, typename Operator>
void forAllEdgesAdjUnionImbalancedSpGEMM(HornetGraph &hornetA,
								   		 HornetGraph &hornetB, 
								   		 // typename HornetGraph::VertexType* queue,
								   		 vid_t* queue,
								   		 const unsigned long long start, 
								   		 const unsigned long long end, 
								   		 const Operator &op,
								    	 unsigned long long threads_per_union, 
								   		 int flag,
								   		 cudaStream_t stream, 
                                         int startRow) {
    unsigned long long size = end - start; // end is exclusive
    auto grid_size = size*threads_per_union;
    auto _size = size;
    while (grid_size > (1ULL<<31)) {
        // FIXME get 1<<31 from Hornet
        _size >>= 1;
        grid_size = _size*threads_per_union;
    }
    if (size == 0)
        return;

    forAllEdgesAdjUnionImbalancedKernelSpGEMM
        <<< xlib::ceil_div<BLOCK_SIZE_OP2>(grid_size), BLOCK_SIZE_OP2, 0, stream >>>
        (hornetA->device(), hornetB->device(), queue, start, end, threads_per_union, flag, op, startRow);
    CHECK_CUDA_ERROR
}


struct OPERATOR_AdjIntersectionCountBalancedSpGEMM {
    triangle_t* d_IntersectPerVertexPair;
    vid_t NV;

    OPERATOR(Vertex &u, Vertex& v, vid_t* ui_begin, vid_t* ui_end, wgt0_t* u_weight, 
            vid_t* vi_begin, vid_t* vi_end, wgt0_t* v_weight,int FLAG, int startRow) {
        int count = 0;
        // wgt0_t count = 0;
        bool vSrc=false;
		if(FLAG&2){
            vSrc=true;
		}
        if((FLAG&1)==0){
            int comp_equals, comp1, comp2, ui_bound, vi_bound;
            int vpos=0, upos=0;
            // printf("Intersecting %d, %d: %d -> %d, %d -> %d\n", u.id(), v.id(), *ui_begin, *ui_end, *vi_begin, *vi_end);
            while (vi_begin <= vi_end && ui_begin <= ui_end) {
                comp_equals = (*ui_begin == *vi_begin);
                // count += comp_equals;
                count += comp_equals*(u_weight[upos]*v_weight[vpos]);
                comp1 = (*ui_begin >= *vi_begin);
                comp2 = (*ui_begin <= *vi_begin);
                ui_bound = (ui_begin == ui_end);
                vi_bound = (vi_begin == vi_end);
                // early termination
                if ((ui_bound && comp2) || (vi_bound && comp1))
                    break;
                if ((comp1 && !vi_bound) || ui_bound){
                    vi_begin += 1;vpos++;
                }
                if ((comp2 && !ui_bound) || vi_bound){
                    ui_begin += 1;upos++;
                }
            }
        } else {
            vid_t vi_low, vi_high, vi_mid;
            int u_pos = 0; // using this field to get index for the weight
            while (ui_begin <= ui_end) {
                auto search_val = *ui_begin;
                vi_low = 0;
                vi_high = vi_end-vi_begin;
                while (vi_low <= vi_high) {
                    vi_mid = (vi_low+vi_high)>>1;
                    auto comp = (*(vi_begin+vi_mid) - search_val);
                    if (!comp) {
                        // count += 1;//u_weight[u_pos]*v_weight[vi_mid];
                        count +=u_weight[u_pos]*v_weight[vi_mid];
                        break;
                    }
                    int geq=comp>0;
                    vi_high = geq*(vi_mid-1)+(1-geq)*vi_high;
                    vi_low = (1-geq)*(vi_mid+1)+(geq)*vi_low;
                    // if (comp > 0) {
                    //     vi_high = vi_mid-1;
                    // } else if (comp < 0) {
                    //     vi_low = vi_mid+1;
                    // }
                }
                ui_begin += 1;
                u_pos++;
            }
        }

        // if (blockIdx.x == 0){
        // 	printf("(u,v) = (%d,%d)", u.id(),v.id());
        // }
        // printf("*");
        if (vSrc == false){ // might need to reorder if and else clauses
    		atomicAdd(d_IntersectPerVertexPair+(u.id()-startRow)*NV+v.id(), count);
		}else{
            atomicAdd(d_IntersectPerVertexPair+(v.id()-startRow)*NV+u.id(), count);
		}
        //atomicAdd(d_triPerVertex+v.id(), count);
    }
};

__global__ void compressNNZtoBatch(triangle_t* d_IntersectCount,
                                   int64_t maxNumberIntersections,
                                   int32_t startRow,
                                   vid_t rowLen,
                                   vid_t *d_batchSizeCounter,
                                   vid_t *d_src,
                                   vid_t *d_dest){
    vid_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >=maxNumberIntersections)
        return;

    if(d_IntersectCount[id]!=0){
        auto pos = atomicAdd(d_batchSizeCounter,1);
        d_src[pos]  = (id/rowLen)+startRow;
        d_dest[pos] = id%rowLen;
    }
}

////adj_unions and bin_edges defined in Operator++i.cuh
//template<typename Operator>
void forAllAdjUnions(HornetGraph*    hornetA,
                     HornetGraph*    hornetB,
                     HornetGraph*    hornetC,
                     int concurrentIntersections,
                     float workFactor,
                     bool sanityCheck){

    //using BinEdges = bin_edges<typename HornetGraph::VertexType>;
    //HostDeviceVar<queue_info<typename HornetGraph::VertexType>> hd_queue_info_spgemm;
    const int MAX_STREAMS = 32;
    cudaStream_t streams[MAX_STREAMS];
    for(int i=0;i<MAX_STREAMS; i++)
      cudaStreamCreate ( &(streams[i]));

    queue_info_spgemm<typename HornetGraph::VertexType> hd_queue_info_spgemm;
    // load_balancing::VertexBased1 load_balancing ( hornetA );

    // memory allocations host and device side
    int64_t maxNumberIntersections = concurrentIntersections;
    // int64_t numRowPartitions =  ((int64_t)hornetA.nV()*(int64_t)hornetB.nV()) / maxNumberIntersections + 
    //                          ((((int64_t)hornetA.nV()*(int64_t)hornetB.nV()) % maxNumberIntersections) ? 1 : 0);
    int64_t numRowSizes = maxNumberIntersections/(int64_t)hornetB->nV();
    int64_t numPartitions =  ((int64_t)hornetA->nV()) / numRowSizes  + 
                            +(((((int64_t)hornetA->nV()) % numRowSizes) ) ? 1 : 0);


    printf("numRowSizes = %ld\n",numRowSizes);
    printf("numPartitions = %ld\n",numPartitions);

    // hornets_nest::gpu::allocate(hd_queue_info_spgemm.d_edge_queue, 2*hornetA->nV()*hornetA.nV());
    // gpu::allocate(d_IntersectCount, hornetA.nV()*hornetA.nV());
    // cudaMemset(d_IntersectCount, 0, hornetA.nV()*hornetA.nV()*sizeof(triangle_t));
    hornets_nest::gpu::allocate(hd_queue_info_spgemm.d_edge_queue, 2*maxNumberIntersections);    
    triangle_t* d_IntersectCount;
    gpu::allocate(d_IntersectCount, maxNumberIntersections);
    
    hornets_nest::gpu::allocate(hd_queue_info_spgemm.d_queue_sizes, MAX_ADJ_UNIONS_BINS);
    unsigned long long *queue_sizes = (unsigned long long *)calloc(MAX_ADJ_UNIONS_BINS, sizeof(unsigned long long));
    hornets_nest::gpu::allocate(hd_queue_info_spgemm.d_queue_pos, MAX_ADJ_UNIONS_BINS+1);
    unsigned long long *queue_pos = (unsigned long long *)calloc(MAX_ADJ_UNIONS_BINS+1, sizeof(unsigned long long));

    vid_t *d_src,*d_dest, *d_batchSizeCounter;

    vid_t h_batchSizeCounter = 0;
    hornets_nest::gpu::allocate(d_src, maxNumberIntersections);
    hornets_nest::gpu::allocate(d_dest, maxNumberIntersections);
    hornets_nest::gpu::allocate(d_batchSizeCounter, 1);
//    hornets_nest::gpu::allocate(d_weight, maxNumberIntersections); // currently disabled since we are not actually computing weights



    int startRow = 0;
    for (int np = 0; np <numPartitions; np++)
    {
        hornets_nest::gpu::memsetZero(hd_queue_info_spgemm.d_queue_sizes, MAX_ADJ_UNIONS_BINS);
        hornets_nest::gpu::memsetZero(hd_queue_info_spgemm.d_queue_pos, MAX_ADJ_UNIONS_BINS+1);
        cudaMemset(d_IntersectCount, 0, maxNumberIntersections*sizeof(triangle_t));


        startRow = np * numRowSizes;
        const int BLOCK_SIZE = 512;
        printf("StartRow = %d\n",startRow);
        // int threadBlocks = hornetA.nV();///BLOCK_SIZE + ((hornetA.nV()%BLOCK_SIZE)?1:0);
        int threadBlocks = numRowSizes;//

        //printf("%d %d\n", dimGrid.x, dimGrid.y);
        //printf("%d %d\n", dimBlock.x, dimBlock.y);
        bin_vertex_pair_spgemm<typename HornetGraph::VertexType> <<<threadBlocks,BLOCK_SIZE>>> (hornetA->device(),hornetB->device(),hd_queue_info_spgemm, true, workFactor, startRow);
    	// CHECK_CUDA_ERROR
    	// printf("passed 1");

        hornets_nest::gpu::copyToHost(hd_queue_info_spgemm.d_queue_sizes, MAX_ADJ_UNIONS_BINS, queue_sizes);
        std::partial_sum(queue_sizes, queue_sizes+MAX_ADJ_UNIONS_BINS, queue_pos+1);

        // transfer prefx results to device
        hornets_nest::host::copyToDevice(queue_pos, MAX_ADJ_UNIONS_BINS+1, hd_queue_info_spgemm.d_queue_pos);

        bin_vertex_pair_spgemm<typename HornetGraph::VertexType> <<<threadBlocks,BLOCK_SIZE>>> (hornetA->device(),hornetB->device(),hd_queue_info_spgemm, false, workFactor, startRow);
    	// CHECK_CUDA_ERROR
    	// printf("passed 2");

        const int BALANCED_THREADS_LOGMAX = 31-__builtin_clz(BLOCK_SIZE_OP2)+1; // assumes BLOCK_SIZE is int type
        int bin_index;
        int bin_offset = 0;
        unsigned long long start_index = 0; 
        unsigned long long end_index;
        //ToDo 
        unsigned int threads_per;
        unsigned long long size;
        int threads_log = 1;
        // balanced kernel
        int streamCounter=0;

        if(0){

            while ((threads_log < BALANCED_THREADS_LOGMAX) && (threads_log+LOG_OFFSET_BALANCED < BINS_1D_DIM)) 
            {
                bin_index = bin_offset+(threads_log+LOG_OFFSET_BALANCED)*BINS_1D_DIM;
                end_index = queue_pos[bin_index];
                size = end_index - start_index;
                if (size) {
                    threads_per = 1 << (threads_log-1); 
                    ////ToDo
                     forAllEdgesAdjUnionBalancedSpGEMM(*hornetA, *hornetB, hd_queue_info_spgemm.d_edge_queue, start_index, end_index, 
                                                       OPERATOR_AdjIntersectionCountBalancedSpGEMM {d_IntersectCount, hornetA->nV()}, threads_per, 0,streams[streamCounter++],startRow);
                }
                start_index = end_index;
                threads_log += 1;
            }
            // process remaining "tail" bins
            bin_index = MAX_ADJ_UNIONS_BINS/2;
            end_index = queue_pos[bin_index];
            size = end_index - start_index;
            if (size) {
                threads_per = 1 << (threads_log-1); 
                ////ToDo
                forAllEdgesAdjUnionBalancedSpGEMM(*hornetA, *hornetB, hd_queue_info_spgemm.d_edge_queue, start_index, end_index, 
                                                  OPERATOR_AdjIntersectionCountBalancedSpGEMM {d_IntersectCount, hornetA->nV()}, threads_per, 0,streams[streamCounter++],startRow);
            }
            start_index = end_index;
        }else
        {
            bin_offset = 0;

            streamCounter=0;
            int currIndex = bin_offset;
            for(int bin_r=0; bin_r<BINS_1D_DIM; bin_r++){
                threads_per = 1;
                for(int bin_c=0; bin_c<BINS_1D_DIM; bin_c++){

                    if(bin_r==0 && bin_c==0){
                        size = queue_pos[0]; 
                        end_index = queue_pos[0];
                        start_index = 0;
                    }
                    else{
                        size = queue_pos[currIndex]-queue_pos[currIndex-1]; 
                        end_index = queue_pos[currIndex];
                        start_index = queue_pos[currIndex-1];
                    }

                    threads_per = std::max((bin_r+bin_c)/5,1);
                    if(threads_per<=1)
                        threads_per=1;
                    else
                        threads_per = 1 << (threads_per-1);

                    if (threads_per > 256)
                        threads_per=256;

                    if (size) {
                        ////ToDo
                        forAllEdgesAdjUnionBalancedSpGEMM(*hornetA, *hornetB, 
                                                        hd_queue_info_spgemm.d_edge_queue, 
                                                        start_index, end_index, 
                                                        OPERATOR_AdjIntersectionCountBalancedSpGEMM {d_IntersectCount, hornetA->nV()}, 
                                                        threads_per, 0,streams[streamCounter],startRow);

                        streamCounter++;
                        if(streamCounter>=MAX_STREAMS)
                            streamCounter=0;
                    }
                    currIndex++;
                }
            }
        }

        // imbalanced kernel
        if(0){
            const int IMBALANCED_THREADS_LOGMAX = BINS_1D_DIM-1; 
            bin_offset = MAX_ADJ_UNIONS_BINS/2;
            threads_log = 1;
            threads_per = 1;

            streamCounter=0;
            // printf("\n");
            while ((threads_log < IMBALANCED_THREADS_LOGMAX) && (threads_log+LOG_OFFSET_IMBALANCED < BINS_1D_DIM)) 
            {
                bin_index = bin_offset+(threads_log+LOG_OFFSET_IMBALANCED)*BINS_1D_DIM;
                end_index = queue_pos[bin_index];
                size = end_index - start_index;
                // printf("%llu \n", size);
                if (size ) {
                    // threads_per = 1 << (threads_log-1);
                    //triangle_t* tempstupid;
                    ////ToDo 
                    forAllEdgesAdjUnionImbalancedSpGEMM(hornetA, hornetB, 
                                                 hd_queue_info_spgemm.d_edge_queue,
                                                 start_index, end_index, 
                                                 OPERATOR_AdjIntersectionCountBalancedSpGEMM {d_IntersectCount, hornetA->nV()}, 
                                                 threads_per, 1, streams[streamCounter++],
                                                 startRow);
                }
                start_index = end_index;
                // threads_log += 1;
                threads_log += 1;
                if(threads_log > 3) 
                    threads_per=2;
                if(threads_log > 6)
                    threads_per=8;
                // printf("%lld\n",size);

            }
            // process remaining "tail" bins
            bin_index = MAX_ADJ_UNIONS_BINS;
            end_index = queue_pos[bin_index];
            size = end_index - start_index;
            if (size) {
                threads_per = 1 << (threads_log-1);
                ////ToDo
                forAllEdgesAdjUnionImbalancedSpGEMM(hornetA, hornetB, 
                                                 hd_queue_info_spgemm.d_edge_queue,
                                                 start_index, end_index, 
                                                 OPERATOR_AdjIntersectionCountBalancedSpGEMM {d_IntersectCount, hornetA->nV()}, 
                                                 threads_per, 1, streams[streamCounter++],
                                                 startRow); 
            }

        }else{
            bin_offset = MAX_ADJ_UNIONS_BINS/2;

            streamCounter=0;
            int currIndex = bin_offset;
            for(int bin_r=0; bin_r<BINS_1D_DIM; bin_r++){
                threads_per = 1;
                for(int bin_c=0; bin_c<BINS_1D_DIM; bin_c++){

                    size = queue_pos[currIndex]-queue_pos[currIndex-1]; 
                    end_index = queue_pos[currIndex];
                    start_index = queue_pos[currIndex-1];

                    // threads_per = 1 << (1 + (bin_r+bin_c)/5);
                    // if (threads_per >256)
                    //     threads_per=256;
                    threads_per = std::max((bin_r+bin_c)/5,1);
                    if(threads_per<=1)
                        threads_per=1;
                    else
                        threads_per = 1 << (threads_per-1);

                    if (threads_per > 256)
                        threads_per=256;


                    if (size) {
                        ////ToDo
                        forAllEdgesAdjUnionImbalancedSpGEMM(hornetA, hornetB, 
                                                         hd_queue_info_spgemm.d_edge_queue,
                                                         start_index, end_index, 
                                                         OPERATOR_AdjIntersectionCountBalancedSpGEMM {d_IntersectCount, hornetA->nV()}, 
                                                         threads_per, 1, streams[streamCounter],startRow); 
                        streamCounter++;
                    	if(streamCounter>=MAX_STREAMS)
                            streamCounter=0;
                    }
                    currIndex++;
                }

            }
        }
        cudaMemset(d_batchSizeCounter,0,sizeof(vid_t));


        int64_t compressTB = maxNumberIntersections/BLOCK_SIZE + ((maxNumberIntersections%BLOCK_SIZE)?1:0);

        compressNNZtoBatch<<<compressTB,BLOCK_SIZE>>>(d_IntersectCount, maxNumberIntersections, startRow, hornetB->nV() ,d_batchSizeCounter,d_src,d_dest);

        cudaMemcpy(&h_batchSizeCounter, d_batchSizeCounter, sizeof(vid_t), cudaMemcpyDeviceToHost);

        if(h_batchSizeCounter>0){
            UpdatePtr ptr(h_batchSizeCounter, d_src, d_dest,NULL);
            Update batch_update(ptr);
            hornetC->insert(batch_update,false,false);

        }

    }

    hornets_nest::gpu::free(d_batchSizeCounter);
    hornets_nest::gpu::free(d_src);
    hornets_nest::gpu::free(d_dest);
    hornets_nest::gpu::free(hd_queue_info_spgemm.d_queue_pos); 
    hornets_nest::gpu::free(hd_queue_info_spgemm.d_queue_sizes);
    hornets_nest::gpu::free(hd_queue_info_spgemm.d_edge_queue);
    free(queue_sizes);
    free(queue_pos);
    
    if (sanityCheck){
        cudaDeviceSynchronize();
	    triangle_t* h_IntersectCount;
	    host::allocate(h_IntersectCount, hornetA->nV()*hornetA->nV());
	    cudaMemcpy(h_IntersectCount, d_IntersectCount, hornetA->nV()*hornetA->nV()*sizeof (triangle_t), cudaMemcpyDeviceToHost);
	    int NNZ=0;
	    triangle_t sum_row=0;
	    for (int i=0; i<hornetA->nV()*hornetA->nV(); i++){
	    	if (i<hornetA->nV()){
	    		printf ("%d", h_IntersectCount[i]);
	    		sum_row += h_IntersectCount[i];
	    	}
	        if (h_IntersectCount[i] != 0){
	        	// printf("i is = %d \n", i);
	        	NNZ++;
	        }
	    }
	    printf("sum_row = %d \n", sum_row);
	    cudaDeviceSynchronize();
	    printf("NNZ = %d \n", NNZ);

	    host::free(h_IntersectCount);
	}
    gpu::free(d_IntersectCount);


    for(int i=0;i<MAX_STREAMS; i++)
      cudaStreamDestroy ( streams[i]);
}

void SpGEMM::reset(){

}

void SpGEMM::run() {
    forAllAdjUnions(hornetA, hornetB, hornetC, concurrentIntersections, workFactor, sanityCheck);
}

// void SpGEMM::run(const int WORK_FACTOR=1){
// }

// template<typename HornetGraph>
// void SpGEMM<HornetGraph>::release(){
void SpGEMM::release(){
    //printf("Inside release\n");
    gpu::free(vertex_pairs);
    vertex_pairs = nullptr;
}

// template<typename HornetGraph>
//void SpGEMM<HornetGraph>::init(){
void SpGEMM::init(){
    //printf("Inside init. Printing hornet.nV(): %d\n", hornet.nV());
    //gpu::allocate(triPerVertex, hornet.nV());
    reset();
}

} // namespace hornets_nest