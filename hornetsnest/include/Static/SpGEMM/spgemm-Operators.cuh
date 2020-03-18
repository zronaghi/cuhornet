#pragma once

#include <Device/Util/Timer.cuh>
#include <Operator++.cuh>
#include <StandardAPI.hpp>

#include "Static/SpGEMM/spgemm.cuh"

namespace hornets_nest {
//namespace detail {


// template<typename HornetDevice, typename T, typename Operator>
// __global__ void forAllVertexPairsKernel(HornetDevice hornet, T* __restrict__ array, int size, Operator op) {
//     int     id = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
//     for (auto i = id; i < size; i += stride) {
//         auto v1_id = array[i].x;
//         auto v2_id = array[i].y;
//         auto v1 = hornet.vertex(v1_id);
//         auto v2 = hornet.vertex(v2_id);
//         op(v1, v2);
//     }
// }

// template<typename HornetDevice, typename T, typename Operator, typename vid_t>
// __global__ void forAllEdgesAdjUnionSequentialKernel(HornetDevice hornet, T* __restrict__ array, unsigned long long size, Operator op, int flag) {
//     int     id = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
//     for (auto i = id; i < size; i += stride) {
//         auto src_vtx = hornet.vertex(array[2*i]);
//         auto dst_vtx = hornet.vertex(array[2*i+1]);
//         degree_t src_deg = src_vtx.degree();
//         degree_t dst_deg = dst_vtx.degree();
//         vid_t* src_begin = src_vtx.neighbor_ptr();
//         vid_t* dst_begin = dst_vtx.neighbor_ptr();
//         vid_t* src_end = src_begin+src_deg-1;
//         vid_t* dst_end = dst_begin+dst_deg-1;
//         op(src_vtx, dst_vtx, src_begin, src_end, dst_begin, dst_end, flag);
//     }
// }

// namespace adj_union {
    
//     template <typename vid_t>
//     __device__ __forceinline__
//     void bSearchPath(vid_t* u, vid_t *v, int u_len, int v_len, 
//                      vid_t low_vi, vid_t low_ui, 
//                      vid_t high_vi, vid_t high_ui, 
//                      vid_t* curr_vi, vid_t* curr_ui) {
//         vid_t mid_ui, mid_vi;
//         int comp1, comp2, comp3;
//         while (1) {
//             mid_ui = (low_ui+high_ui)/2;
//             mid_vi = (low_vi+high_vi+1)/2;

//             comp1 = (u[mid_ui] < v[mid_vi]);
            
//             if (low_ui == high_ui && low_vi == high_vi) {
//                 *curr_vi = mid_vi;
//                 *curr_ui = mid_ui;
//                 break;
//             }
//             if (!comp1) {
//                 low_ui = mid_ui;
//                 low_vi = mid_vi;
//                 continue;
//             }

//             comp2 = (u[mid_ui+1] >= v[mid_vi-1]);
//             if (comp1 && !comp2) {
//                 high_ui = mid_ui+1;
//                 high_vi = mid_vi-1;
//             } else if (comp1 && comp2) {
//                 comp3 = (u[mid_ui+1] < v[mid_vi]);
//                 *curr_vi = mid_vi-comp3;
//                 *curr_ui = mid_ui+comp3;
//                 break;
//             }
//        }
//     }
// }

template<typename HornetDevice, typename T, typename Operator>
__global__ void forAllEdgesAdjUnionBalancedKernel(HornetDevice hornet, T* __restrict__ array, unsigned long long start, unsigned long long end, unsigned long long threads_per_union, int flag, Operator op) {

    using namespace adj_union;
    using vid_t = typename HornetDevice::VertexType;
    int       id = blockIdx.x * blockDim.x + threadIdx.x;
    int queue_id = id / threads_per_union;
    int thread_union_id = threadIdx.x % threads_per_union;
    int block_local_id = threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int queue_stride = stride / threads_per_union;

    // TODO: dynamic vs. static shared memory allocation?
    __shared__ vid_t pathPoints[256*2]; // i*2+0 = vi, i+2+1 = u_i
    for (auto i = start+queue_id; i < end; i += queue_stride) {
        auto src_vtx = hornet.vertex(array[2*i]);
        auto dst_vtx = hornet.vertex(array[2*i+1]);
        int srcLen = src_vtx.degree();
        int destLen = dst_vtx.degree();
        int total_work = srcLen + destLen - 1;
        vid_t src = src_vtx.id();
        vid_t dest = dst_vtx.id();

        bool avoidCalc = (src == dest) || (srcLen < 2);
        if (avoidCalc)
            continue;

        // determine u,v where |adj(u)| <= |adj(v)|
        bool sourceSmaller = srcLen < destLen;
        vid_t u = sourceSmaller ? src : dest;
        vid_t v = sourceSmaller ? dest : src;
        auto u_vtx = sourceSmaller ? src_vtx : dst_vtx;
        auto v_vtx = sourceSmaller ? dst_vtx : src_vtx;
        degree_t u_len = sourceSmaller ? srcLen : destLen;
        degree_t v_len = sourceSmaller ? destLen : srcLen;
        vid_t* u_nodes = hornet.vertex(u).neighbor_ptr();
        vid_t* v_nodes = hornet.vertex(v).neighbor_ptr();

        int work_per_thread = total_work/threads_per_union;
        int remainder_work = total_work % threads_per_union;
        int diag_id, next_diag_id;
        diag_id = thread_union_id*work_per_thread + std::min(thread_union_id, remainder_work);
        next_diag_id = (thread_union_id+1)*work_per_thread + std::min(thread_union_id+1, remainder_work);
        vid_t low_ui, low_vi, high_vi, high_ui, ui_curr, vi_curr;
        if (diag_id > 0 && diag_id < total_work) {
            if (diag_id < u_len) {
                low_ui = diag_id-1;
                high_ui = 0;
                low_vi = 0;
                high_vi = diag_id-1;
            } else if (diag_id < v_len) {
                low_ui = u_len-1;
                high_ui = 0;
                low_vi = diag_id-u_len;
                high_vi = diag_id-1;
            } else {
                low_ui = u_len-1;
                high_ui = diag_id - v_len;
                low_vi = diag_id-u_len;
                high_vi = v_len-1;
            }
            bSearchPath(u_nodes, v_nodes, u_len, v_len, low_vi, low_ui, high_vi,
                     high_ui, &vi_curr, &ui_curr);
            pathPoints[block_local_id*2] = vi_curr; 
            pathPoints[block_local_id*2+1] = ui_curr; 
        }

        __syncthreads();

        vid_t vi_begin, ui_begin, vi_end, ui_end;
        vi_begin = ui_begin = vi_end = ui_end = -1;
        int vi_inBounds, ui_inBounds;
        if (diag_id == 0) {
            vi_begin = 0;
            ui_begin = 0;
        } else if (diag_id > 0 && diag_id < total_work) {
            vi_begin = vi_curr;
            ui_begin = ui_curr;
            vi_inBounds = (vi_curr < v_len-1);
            ui_inBounds = (ui_curr < u_len-1);
            if (vi_inBounds && ui_inBounds) {
                int comp = (u_nodes[ui_curr+1] >= v_nodes[vi_curr+1]);
                vi_begin += comp;
                ui_begin += !comp;
            } else {
                vi_begin += vi_inBounds;
                ui_begin += ui_inBounds;
            }
        }
        
        if ((diag_id < total_work) && (next_diag_id >= total_work)) {
            vi_end = v_len - 1;
            ui_end = u_len - 1;
        } else if (diag_id < total_work) {
            vi_end = pathPoints[(block_local_id+1)*2];
            ui_end = pathPoints[(block_local_id+1)*2+1];
        }
        if (diag_id < total_work) {
            op(u_vtx, v_vtx, u_nodes+ui_begin, u_nodes+ui_end, v_nodes+vi_begin, v_nodes+vi_end, flag);
        }
    }
}



} // namespace hornets_nest
