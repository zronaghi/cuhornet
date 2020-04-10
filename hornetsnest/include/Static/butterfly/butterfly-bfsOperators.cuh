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

    vert_t src = bfs().d_lrbRelabled[k];
    degree_t currLevel = bfs().currLevel;
    vert_t lower = bfs().lower;
    vert_t upper = bfs().upper;

    vert_t* neighPtr = hornet.vertex(src).neighbor_ptr();
    int length = hornet.vertex(src).degree();

    for (int i=0; i<length; i++) {
       vert_t dst_id = neighPtr[i]; 


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
  int N,
  int start){
    int k = blockIdx.x;
    int tid = threadIdx.x;
    if(k>=N){
        printf("should never happen\n");
        return;
    }
    k+=start;    

    vert_t src = bfs().d_lrbRelabled[k];
    degree_t currLevel = bfs().currLevel;
    vert_t lower = bfs().lower;
    vert_t upper = bfs().upper;

    vert_t* neighPtr = hornet.vertex(src).neighbor_ptr();
    int length = hornet.vertex(src).degree();

    for (int i=tid; i<length; i+=blockDim.x) {
       vert_t dst_id = neighPtr[i]; 

        if(bfs().d_dist[dst_id]==INT32_MAX){

            // degree_t prev = atomicCAS(bfs().d_dist + dst_id, INT32_MAX, currLevel);
            degree_t prev = atomicMin(bfs().d_dist + dst_id, currLevel);

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
__global__ void BFSTopDown_One_Iter_kernel__extra_fat(
  HornetDevice hornet , 
  HostDeviceVar<butterflyData> bfs, 
  int N){
    int k=0;
    int tid = threadIdx.x + blockIdx.x *blockDim.x;
    int stride = blockDim.x*gridDim.x;

    degree_t currLevel = bfs().currLevel;
    vert_t lower = bfs().lower;
    vert_t upper = bfs().upper;

    while (k<N)
    {
        vert_t src = bfs().d_lrbRelabled[k];

        vert_t* neighPtr = hornet.vertex(src).neighbor_ptr();
        int length = hornet.vertex(src).degree();

        for (int i=tid; i<length; i+=stride) {
           vert_t dst_id = neighPtr[i]; 

            if(bfs().d_dist[dst_id]==INT32_MAX){

                // degree_t prev = atomicCAS(bfs().d_dist + dst_id, INT32_MAX, currLevel);
                degree_t prev = atomicMin(bfs().d_dist + dst_id, currLevel);

                if (prev == INT32_MAX){

                    if (dst_id >= lower && dst_id < upper){

                            // printf("%d ",dst_id);
                        bfs().queueLocal.insert(dst_id);
                    }
                    bfs().queueRemote.insert(dst_id);
                }
            }
        }

        k++;

    }

}

template<bool sorted, typename HornetDevice>
__global__ void NeighborUpdates_QueueingKernel(
  HornetDevice hornet , 
  HostDeviceVar<butterflyData> bfs, 
  int N,
  degree_t currLevel,
  vert_t lower,
  vert_t upper){
    int k = threadIdx.x + blockIdx.x *blockDim.x;
    if(k>=N)
        return;


    vert_t dst = bfs().d_buffer[k];
    // degree_t currLevel = bfs().currLevel;
    // vert_t lower = bfs().lower;
    // vert_t upper = bfs().upper;

    // if(sorted==true){
    //     if (k>0){
    //         vert_t dstPrev = bfs().d_buffer[k-1];
    //         if(dstPrev==dst)
    //             return;
    //     }        
    // }


    // degree_t prev = atomicMin(bfs().d_dist + dst, currLevel);

    // if(prev == INT32_MAX){
    //     bfs().queueRemote.insert(dst);

    //     if (dst >= bfs().lower && dst <bfs().upper){
    //         bfs().queueLocal.insert(dst);
    //     }
    // }



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




struct BFSTopDown_One_Iter {
    HostDeviceVar<butterflyData> bfs;

    OPERATOR(Vertex& src, Edge& edge){

        vert_t dst_id = edge.dst_id();        
        degree_t currLevel = bfs().currLevel;
        vert_t lower = bfs().lower;
        vert_t upper = bfs().upper;

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
        vert_t dst = dst_v.id();
        degree_t currLevel = bfs().currLevel;

        // if (bfs().d_dist[dst] == INT32_MAX){
            // degree_t prev = atomicCAS(bfs().d_dist + dst, INT32_MAX, currLevel);
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
        // }
    }
};



} // namespace hornets_nest
