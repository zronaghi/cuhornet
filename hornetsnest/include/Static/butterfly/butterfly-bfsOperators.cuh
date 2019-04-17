
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



// // Used at the very beginning of every BC computation.
// // Once per root
// struct InitOneTree {
//     HostDeviceVar<BCData> bcd;

//     // Used at the very beginning
//     OPERATOR(vid_t src) {
//         bcd().d[src] = INT32_MAX;
//         bcd().sigma[src] = 0;
//         bcd().delta[src] = 0.0;
//     }
// };

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
