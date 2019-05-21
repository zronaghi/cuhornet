#pragma once

#include "Static/butterfly/butterfly-bfs.cuh"

namespace hornets_nest {


// Used at the very beginning of every BC computation.
// Used only once.
struct InitBFS {
    HostDeviceVar<butterflyData> bfs;


    OPERATOR(vid_t src) {
        bfs().d_Marked[src] = false;
        bfs().d_dist[src] = INT32_MAX;
    }
};




template<typename HornetDevice>
__global__ void BFSTopDown_One_Iter_kernel(
  HornetDevice hornet , 
  HostDeviceVar<butterflyData> bfs, 
  int N,
  int start){
    int k = threadIdx.x + blockIdx.x *blockDim.x;
    if(k>=N)
        return;
    k+=start;

    vid_t src = bfs().d_lrbRelabled[k];
    degree_t currLevel = bfs().currLevel;
    vid_t lower = bfs().lower;
    vid_t upper = bfs().upper;

    vid_t* neighPtr = hornet.vertex(src).neighbor_ptr();
    int length = hornet.vertex(src).degree();

    for (int i=0; i<length; i++) {
       vid_t dst_id = neighPtr[i]; 


        if(bfs().d_dist[dst_id]==INT32_MAX){

            degree_t prev = atomicCAS(bfs().d_dist + dst_id, INT32_MAX, currLevel);

            if (prev == INT32_MAX){

                if (dst_id >= lower && dst_id < upper){

                        // printf("%d ",dst_id);
                    bfs().queueLocal.insert(dst_id);
                }

                bfs().queueRemote.insert(dst_id);
            }


        }


    }
}

template<typename HornetDevice>
__global__ void BFSTopDown_One_Iter_kernel_fat(
  HornetDevice hornet , 
  HostDeviceVar<butterflyData> bfs, 
  int N){
    int k = blockIdx.x;
    int tid = threadIdx.x;
    if(k>=N){
        printf("should never happen\n");
        return;
    }

    vid_t src = bfs().d_lrbRelabled[k];
    degree_t currLevel = bfs().currLevel;
    vid_t lower = bfs().lower;
    vid_t upper = bfs().upper;

    vid_t* neighPtr = hornet.vertex(src).neighbor_ptr();
    int length = hornet.vertex(src).degree();

    for (int i=tid; i<length; i+=blockDim.x) {
       vid_t dst_id = neighPtr[i]; 


        if(bfs().d_dist[dst_id]==INT32_MAX){

            degree_t prev = atomicCAS(bfs().d_dist + dst_id, INT32_MAX, currLevel);

            if (prev == INT32_MAX){

                if (dst_id >= lower && dst_id < upper){

                        // printf("%d ",dst_id);
                    bfs().queueLocal.insert(dst_id);
                }

                bfs().queueRemote.insert(dst_id);
            }


        }


    }
}





struct BFSTopDown_One_Iter {
    HostDeviceVar<butterflyData> bfs;

    OPERATOR(Vertex& src, Edge& edge){

        vid_t dst_id = edge.dst_id();        
        degree_t currLevel = bfs().currLevel;
        vid_t lower = bfs().lower;
        vid_t upper = bfs().upper;

        if(bfs().d_dist[dst_id]==INT32_MAX){

            degree_t prev = atomicCAS(bfs().d_dist + dst_id, INT32_MAX, currLevel);

            if (prev == INT32_MAX){


                // if(threadIdx.x==0 && blockIdx.x==0){
                //     printf("bfs().lower = %d   && bfs().upper = %d\n",bfs().lower,bfs().upper);
                // }

                // printf("%d,",dst_id);
                // bfs().d_Marked[dst_id ]=true;
                if (dst_id >= lower && dst_id < upper){

                        // printf("%d ",dst_id);
                    bfs().queueLocal.insert(dst_id);
                }

                bfs().queueRemote.insert(dst_id);


                // printf("%d," ,bfs().queueRemoteSize);
                // degree_t temp = (bfs().queueRemoteSize);
                // degree_t pos = atomicAdd(&temp,1);
                // bfs().queueRemote[pos] = dst_id;

            }


        }
    }
};


struct NeighborUpdates {
    HostDeviceVar<butterflyData> bfs;

    OPERATOR(Vertex& dst_v){
        vid_t dst = dst_v.id();
        degree_t currLevel = bfs().currLevel;

        if (bfs().d_dist[dst] == INT32_MAX){
            degree_t prev = atomicCAS(bfs().d_dist + dst, INT32_MAX, currLevel);

            if(prev == INT32_MAX){

                bfs().d_dist[dst]=bfs().currLevel;
                if (dst >= bfs().lower && dst <bfs().upper){
                    // printf("*");
                    bfs().queueLocal.insert(dst);

            // if(bfs().currLevel==0 && bfs().gpu_id==0)                
                    // printf("%d ",dst);

                }
                bfs().queueRemote.insert(dst);
            }
        }
    }
};



} // namespace hornets_nest
