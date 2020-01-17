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

namespace hornets_nest {

    // template<typename HornetGraph>
    // SpGEMM<HornetGraph>::SpGEMM(HornetGraph& hornetA, HornetGraph& hornetB):hornetA(hornetA) ,hornetB(hornetB) {}
    SpGEMM::SpGEMM(HornetGraph& hornetA, HornetGraph& hornetB, HornetGraph& hornetC):hornetA(hornetA), hornetB(hornetB), hornetC(hornetC) {
    
    //gpu allocate to invoke RMM_ALLOC?

    gpu::allocate(vertex_pairs, hornetC.nV());

    reset();

}
    // template<typename HornetGraph>
    // SpGEMM<HornetGraph>::~SpGEMM(){}
    SpGEMM::~SpGEMM(){
        release();
    }
// struct OPERATOR_InitTriangleCounts {
//     triangle_t *d_triPerVertex;

//     OPERATOR (Vertex &vertex) {
//         d_triPerVertex[vertex.id()] = 0;
//     }
// };

/*
 * Naive intersection operator
 * Assumption: access to entire adjacencies of v1 and v2 required
 */
// struct OPERATOR_AdjIntersectionCount {
//     triangle_t* d_triPerVertex;

//     OPERATOR(Vertex& v1, Vertex& v2, int flag) {
//         triangle_t count = 0;
//         int deg1 = v1.degree();
//         int deg2 = v2.degree();
//         vid_t* ui_begin = v1.neighbor_ptr();
//         vid_t* vi_begin = v2.neighbor_ptr();
//         vid_t* ui_end = ui_begin+deg1-1;
//         vid_t* vi_end = vi_begin+deg2-1;
//         int comp_equals, comp1, comp2;
//         while (vi_begin <= vi_end && ui_begin <= ui_end) {
//             comp_equals = (*ui_begin == *vi_begin);
//             count += comp_equals;
//             comp1 = (*ui_begin >= *vi_begin);
//             comp2 = (*ui_begin <= *vi_begin);
//             vi_begin += comp1;
//             ui_begin += comp2;
//             // early termination
//             if ((vi_begin > vi_end) || (ui_begin > ui_end))
//                 break;
//         }
//         atomicAdd(d_triPerVertex+v1.id(), count);
//         atomicAdd(d_triPerVertex+v2.id(), count);
//     }
// };


// struct OPERATOR_AdjIntersectionCountBalanced {
//     triangle_t* d_triPerVertex;

//     OPERATOR(Vertex &u, Vertex& v, vid_t* ui_begin, vid_t* ui_end, vid_t* vi_begin, vid_t* vi_end, int FLAG) {
//         int count = 0;
//         if (!FLAG) {
//             int comp_equals, comp1, comp2, ui_bound, vi_bound;
//             //printf("Intersecting %d, %d: %d -> %d, %d -> %d\n", u.id(), v.id(), *ui_begin, *ui_end, *vi_begin, *vi_end);
//             while (vi_begin <= vi_end && ui_begin <= ui_end) {
//                 comp_equals = (*ui_begin == *vi_begin);
//                 count += comp_equals;
//                 comp1 = (*ui_begin >= *vi_begin);
//                 comp2 = (*ui_begin <= *vi_begin);
//                 ui_bound = (ui_begin == ui_end);
//                 vi_bound = (vi_begin == vi_end);
//                 // early termination
//                 if ((ui_bound && comp2) || (vi_bound && comp1))
//                     break;
//                 if ((comp1 && !vi_bound) || ui_bound)
//                     vi_begin += 1;
//                 if ((comp2 && !ui_bound) || vi_bound)
//                     ui_begin += 1;
//             }
//         } else {
//             vid_t vi_low, vi_high, vi_mid;
//             while (ui_begin <= ui_end) {
//                 auto search_val = *ui_begin;
//                 vi_low = 0;
//                 vi_high = vi_end-vi_begin;
//                 while (vi_low <= vi_high) {
//                     vi_mid = (vi_low+vi_high)/2;
//                     auto comp = (*(vi_begin+vi_mid) - search_val);
//                     if (!comp) {
//                         count += 1;
//                         break;
//                     }
//                     if (comp > 0) {
//                         vi_high = vi_mid-1;
//                     } else if (comp < 0) {
//                         vi_low = vi_mid+1;
//                     }
//                 }
//                 ui_begin += 1;
//             }
//         }

//         atomicAdd(d_triPerVertex+u.id(), count);
//         //atomicAdd(d_triPerVertex+v.id(), count);
//     }
// };

// void SpGEMM::copyTCToHost(triangle_t* h_tcs) {
//     gpu::copyToHost(triPerVertex, hornet.nV(), h_tcs);
// }

// triangle_t SpGEMM::countTriangles(){

//     triangle_t* h_triPerVertex;
//     host::allocate(h_triPerVertex, hornet.nV());
//     gpu::copyToHost(triPerVertex, hornet.nV(), h_triPerVertex);
//     triangle_t sum=0;
//     for(int i=0; i<hornet.nV(); i++){
//         // printf("%d %ld\n", i,outputArray[i]);
//         sum+=h_triPerVertex[i];
//     }
//     free(h_triPerVertex);
//     //triangle_t sum=gpu::reduce(hd_triangleData().triPerVertex, hd_triangleData().nv+1);

//     return sum;
//}

__global__ void CreateIndexPair (vid_t rowc,
                                 vid_t colc,
                                 vid2_t* vertex_pairs) { 

vid_t row = blockIdx.y * blockDim.y + threadIdx.y;
vid_t col = blockIdx.x * blockDim.x + threadIdx.x;
if (row >= rowc || col >= colc) return;

vertex_pairs [row*colc+col].x = row;
vertex_pairs [row*colc+col].y = col;
}

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
                                        bool countOnly, const int WORK_FACTOR) {

        int bin_index;
        vid_t row = blockIdx.y * blockDim.y + threadIdx.y;
        vid_t col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= hornetDeviceA.nV() || col >= hornetDeviceB.nV()) return;
        //     // Choose the bin to place this edge into
        degree_t row_len = hornetDeviceA.vertex(row).degree();
        degree_t col_len = hornetDeviceB.vertex(col).degree();

        //degree_t u_len = (row_len <= col_len) ? src_len : dst_len;
        //degree_t v_len = (row_len <= col_len) ? dst_len : src_len;

        degree_t u_len = row_len;
        degree_t v_len = col_len;
        unsigned int log_v = std::min(32-__clz(v_len), 31);
        unsigned int log_u = std::min(32-__clz(u_len), 31);
        int binary_work_est = u_len*log_v;
        int intersect_work_est = u_len + v_len + log_u;
        int METHOD = ((WORK_FACTOR*intersect_work_est >= binary_work_est)); 
        if (!METHOD && u_len <= 1) {
            bin_index = (METHOD*MAX_ADJ_UNIONS_BINS/2);
        } else if (!METHOD) {
            bin_index = (METHOD*MAX_ADJ_UNIONS_BINS/2)+(log_v*BINS_1D_DIM+log_u);
        } else {
            bin_index = (METHOD*MAX_ADJ_UNIONS_BINS/2)+(log_u*BINS_1D_DIM+log_v); 
        }
        // Either count or add the item to the appropriate queue position
        if (countOnly)
            atomicAdd(&(d_queue_info.ptr()->d_queue_sizes[bin_index]), 1ULL);
        else {
            unsigned long long id = atomicAdd(&(d_queue_info.ptr()->d_queue_pos[bin_index]), 1ULL);
            d_queue_info.ptr()->d_edge_queue[id*2] = row_len.id();
            d_queue_info.ptr()->d_edge_queue[id*2+1] = col_len.id();
    }
}

////ToDo: rewrite the function
// template <typename vid_t>
// void forAllEdgesAdjUnionImbalanced(HornetGraph &hornetA, HornetGraph &hornetB, vid_t queue, const unsigned long long start, const unsigned long long end, const Operator &op, unsigned long long threads_per_union, int flag) {
//     unsigned long long size = end - start; // end is exclusive
//     auto grid_size = size*threads_per_union;
//     auto _size = size;
//     while (grid_size > (1ULL<<31)) {
//         // FIXME get 1<<31 from Hornet
//         _size >>= 1;
//         grid_size = _size*threads_per_union;
//     }
//     if (size == 0)
//         return;
//     detail::forAllEdgesAdjUnionImbalancedKernel
//         <<< xlib::ceil_div<BLOCK_SIZE_OP2>(grid_size), BLOCK_SIZE_OP2 >>>
//         (hornet.device(), queue, start, end, threads_per_union, flag, op);
//     CHECK_CUDA_ERROR
// }

////adj_unions and bin_edges defined in Operator++i.cuh
void forAllAdjUnions(HornetGraph&    hornetA,
                     HornetGraph&    hornetB,
                     HornetGraph&    hornetC){

    //using BinEdges = bin_edges<typename HornetGraph::VertexType>;
    //HostDeviceVar<queue_info<typename HornetGraph::VertexType>> hd_queue_info_spgemm;
    queue_info_spgemm<HornetGraph::VertexType> hd_queue_info_spgemm;
    load_balancing::VertexBased1 load_balancing ( hornetA );

    // memory allocations host and device side
    hornets_nest::gpu::allocate(hd_queue_info_spgemm.d_edge_queue, 2*hornetA.nE());
    hornets_nest::gpu::allocate(hd_queue_info_spgemm.d_queue_sizes, MAX_ADJ_UNIONS_BINS);
    hornets_nest::gpu::memsetZero(hd_queue_info_spgemm.d_queue_sizes, MAX_ADJ_UNIONS_BINS);
    unsigned long long *queue_sizes = (unsigned long long *)calloc(MAX_ADJ_UNIONS_BINS, sizeof(unsigned long long));
    hornets_nest::gpu::allocate(hd_queue_info_spgemm.d_queue_pos, MAX_ADJ_UNIONS_BINS+1);
    hornets_nest::gpu::memsetZero(hd_queue_info_spgemm.d_queue_pos, MAX_ADJ_UNIONS_BINS+1);
    unsigned long long *queue_pos = (unsigned long long *)calloc(MAX_ADJ_UNIONS_BINS+1, sizeof(unsigned long long));

    // figure out cutoffs/counts per bin
    // if (vertex_pairs.size())
    //     forAllVertexPairs(hornet, vertex_pairs, BinEdges {hd_queue_info_spgemm, true, WORK_FACTOR});
    // else
    //forAllEdgeVertexPairs(hornet, BinEdges {hd_queue_info_spgemm, true, WORK_FACTOR}, load_balancing);
    const int BLOCK_SIZE = 32;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(hornetB.nV()/dimBlock.x + (hornetB.nV()%dimBlock.x)?1:0, hornetA.nV()/dimBlock.y + (hornetA.nV()%dimBlock.y)?1:0);
    bin_vertex_pair_spgemm<HornetGraph::VertexType> <<<dimGrid,dimBlock>>> (hornetA.device(),hornetB.device(),hd_queue_info_spgemm, true, 100);

    // copy queue size info to from device to host
    hornets_nest::gpu::copyToHost(hd_queue_info_spgemm.d_queue_sizes, MAX_ADJ_UNIONS_BINS, queue_sizes);
    // prefix sum over bin sizes
    std::partial_sum(queue_sizes, queue_sizes+MAX_ADJ_UNIONS_BINS, queue_pos+1);
    // transfer prefx results to device
    hornets_nest::host::copyToDevice(queue_pos, MAX_ADJ_UNIONS_BINS+1, hd_queue_info_spgemm.d_queue_pos);
    // bin edges
    // if (vertex_pairs.size())
    //     forAllVertexPairs(hornet, vertex_pairs, BinEdges {hd_queue_info_spgemm, false, WORK_FACTOR});
    // else
    //forAllEdgeVertexPairs(hornet, BinEdges {hd_queue_info_spgemm, false, WORK_FACTOR}, load_balancing);
    bin_vertex_pair_spgemm<HornetGraph::VertexType> <<<1,1>>> (hornetA.device(),hornetB.device(),hd_queue_info_spgemm, false, 100);


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
    while ((threads_log < BALANCED_THREADS_LOGMAX) && (threads_log+LOG_OFFSET_BALANCED < BINS_1D_DIM)) 
    {
        bin_index = bin_offset+(threads_log+LOG_OFFSET_BALANCED)*BINS_1D_DIM;
        end_index = queue_pos[bin_index];
        size = end_index - start_index;
        if (size) {
            threads_per = 1 << (threads_log-1); 
            ////ToDo
            //forAllEdgesAdjUnionBalanced(hornet, hd_queue_info_spgemm.d_edge_queue, start_index, end_index, op, threads_per, 0);
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
        //forAllEdgesAdjUnionBalanced(hornet, hd_queue_info_spgemm().d_edge_queue, start_index, end_index, op, threads_per, 0);
    }
    start_index = end_index;

    // imbalanced kernel
    const int IMBALANCED_THREADS_LOGMAX = BINS_1D_DIM-1; 
    bin_offset = MAX_ADJ_UNIONS_BINS/2;
    threads_log = 1;
    while ((threads_log < IMBALANCED_THREADS_LOGMAX) && (threads_log+LOG_OFFSET_IMBALANCED < BINS_1D_DIM)) 
    {
        bin_index = bin_offset+(threads_log+LOG_OFFSET_IMBALANCED)*BINS_1D_DIM;
        end_index = queue_pos[bin_index];
        size = end_index - start_index;
        if (size) {
            threads_per = 1 << (threads_log-1);
            ////ToDo 
            //forAllEdgesAdjUnionImbalanced(hornet, hd_queue_info_spgemm().d_edge_queue, start_index, end_index, op, threads_per, 1);
        }
        start_index = end_index;
        threads_log += 1;
    }
    // process remaining "tail" bins
    bin_index = MAX_ADJ_UNIONS_BINS;
    end_index = queue_pos[bin_index];
    size = end_index - start_index;
    if (size) {
        threads_per = 1 << (threads_log-1);
        ////ToDo 
        //forAllEdgesAdjUnionImbalanced(hornet, hd_queue_info_spgemm().d_edge_queue, start_index, end_index, op, threads_per, 1);
    }

    hornets_nest::gpu::free(hd_queue_info_spgemm.d_queue_pos, hd_queue_info_spgemm.d_queue_sizes, hd_queue_info_spgemm.d_edge_queue);

    threads_per = threads_per+1;
    free(queue_sizes);
    free(queue_pos);
}

void SpGEMMfunc (HornetGraph&    hornetA,
                 HornetGraph&    hornetB,
                 HornetGraph&    hornetC) {


}
// template<typename HornetGraph>
//void SpGEMM<HornetGraph>::reset(){
void SpGEMM::reset(){
    //printf("Inside reset()\n");
    //forAllVertices(hornet, OPERATOR_InitTriangleCounts { triPerVertex });
}

// template<typename HornetGraph>
//void SpGEMM<HornetGraph>::run() {
void SpGEMM::run() {
    //printf("Inside run()\n");
    SpGEMMfunc(hornetA, hornetB, hornetC);
    //forAllAdjUnions(hornet, OPERATOR_AdjIntersectionCount { triPerVertex });
}

// template<typename HornetGraph>
//void SpGEMM<HornetGraph>::run(const int WORK_FACTOR=1){
void SpGEMM::run(const int WORK_FACTOR=1){
    //forAllAdjUnions(hornet, OPERATOR_AdjIntersectionCountBalanced { triPerVertex }, WORK_FACTOR);
}

// template<typename HornetGraph>
// void SpGEMM<HornetGraph>::release(){
void SpGEMM::release(){
    //printf("Inside release\n");
    gpu::free(triPerVertex);
    triPerVertex = nullptr;
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
