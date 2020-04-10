#pragma once

#include "Static/butterfly/butterfly-bfs.cuh"

namespace hornets_nest {


// Used at the very beginning of every BC computation.
// Used only once.
struct InitBFS {
    HostDeviceVar<butterflyData> bfs;


    OPERATOR(vert_t src) {
        bfs().d_Marked[src] = false;
        bfs().d_dist[src] = INT32_MAX;
    }
};



template<bool sorted, typename HornetDevice>
__global__ void NeighborUpdates_QueueingKernel(
  HornetDevice hornet , 
  HostDeviceVar<butterflyData> bfs, 
  int N,
  degree_t currLevel,
  vert_t lower,
  vert_t upper,
  vert_t start=0){
    int k = threadIdx.x + blockIdx.x *blockDim.x;
    if(k>=N)
        return;

    vert_t dst = bfs().d_buffer[k+start];

    if(bfs().d_dist[dst] == INT32_MAX){

        degree_t prev = atomicMin(bfs().d_dist + dst, currLevel);
        if(prev == INT32_MAX)
        {
            bfs().queueRemote.insert(dst);
            if (dst >= bfs().lower && dst <bfs().upper){
                bfs().queueLocal.insert(dst);            
            }
        }
    }

}

struct NeighborUpdates {
    HostDeviceVar<butterflyData> bfs;

    OPERATOR(Vertex& dst_v){
        vert_t dst = dst_v.id();

        // if (bfs().d_dist[dst] == INT32_MAX){
            // degree_t prev = atomicCAS(bfs().d_dist + dst, INT32_MAX, currLevel);

        if(bfs().d_dist[dst] == INT32_MAX){

            degree_t currLevel = bfs().currLevel;
            degree_t prev = atomicMin(bfs().d_dist + dst, currLevel);

            if(prev == INT32_MAX){
                bfs().queueRemote.insert(dst);

                // bfs().d_dist[dst]=bfs().currLevel;
                if (dst >= bfs().lower && dst <bfs().upper){
                    // printf("*");
                    bfs().queueLocal.insert(dst);

            // if(bfs().currLevel==0 && bfs().gpu_id==0)                
                    // printf("%d ",dst);

                }
            }
        }
    }
};


struct BFSTopDown_One_Iter {
    HostDeviceVar<butterflyData> bfs;

    OPERATOR(Vertex& src, Edge& edge){

        vert_t dst_id = edge.dst_id();

        if(bfs().d_dist[dst_id]==INT32_MAX){

            degree_t currLevel = bfs().currLevel;

            degree_t prev = atomicCAS(bfs().d_dist + dst_id, INT32_MAX, currLevel);

            if (prev == INT32_MAX){

                vert_t lower = bfs().lower;
                vert_t upper = bfs().upper;

                if (dst_id >= lower && dst_id < upper){

                        // printf("%d ",dst_id);
                    bfs().queueLocal.insert(dst_id);
                }

                bfs().queueRemote.insert(dst_id);
            }
        }
    }
};




} // namespace hornets_nest
