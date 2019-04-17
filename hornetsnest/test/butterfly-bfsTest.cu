
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off

#include <vector>

#include <omp.h>

#include "Static/butterfly/butterfly-bfs.cuh"
#include "Static/butterfly/butterfly-bfsOperators.cuh"

using namespace std;
using namespace graph;
using namespace graph::structure_prop;
using namespace graph::parsing_prop;

#define CHECK_ERROR(str) \
    {cudaError_t err; err = cudaGetLastError(); if(err!=0) {printf("ERROR %s:  %d %s\n", str, err, cudaGetErrorString(err)); fflush(stdout); exit(0);}}

using namespace timer;
using namespace hornets_nest;



// A recursive binary search function for partitioning the vertices.
// Vertices are NOT split amongst the cores\GPUs thus
// we returns the vertex id with the smallest value larger than x (which is the edge partition)
vid_t vertexBinarySearch(const eoff_t *offsets, vid_t l, vid_t r, eoff_t x) 
{ 
    if (r >= l) { 
        vid_t mid = l + (r - l) / 2; 
  
        // If the element is present at the middle itself 
        if (offsets[mid] == x) // perfect load balancing
            return mid; 
  
        // Check left subarray
        if (offsets[mid] > x) 
            return vertexBinarySearch(offsets, l, mid - 1, x); 
        else
        // Check right subarray 
            return vertexBinarySearch(offsets, mid + 1, r, x); 
    } 
  
    // Return the vertex id of the smallest vertex with an offset greater than x. 
    return l; 
} 


int main(int argc, char* argv[]) {

    // GraphStd<vid_t, eoff_t> graph(UNDIRECTED);
    graph::GraphStd<vid_t, eoff_t> graph;
    CommandLineParam cmd(graph, argc, argv,false);
    Timer<DEVICE> TM;

    // int numGPUs=2; int logNumGPUs=1;
     // int numGPUs=4; int logNumGPUs=2;
    // int numGPUs=8; int logNumGPUs=3;
    // int numGPUs=16; int logNumGPUs=4;


    int numGPUs=4; int logNumGPUs=2; int fanout=1;


    vid_t root = graph.max_out_degree_id();
    // if (argc>=3)
    //     root = atoi(argv[2]);

    if (argc>=3){
        numGPUs = atoi(argv[2]);
        logNumGPUs = atoi(argv[3]);
    }
    if (argc>=5){
        fanout = atoi(argv[4]);
    }

    // if (argc>=4)
    //     numGPUs = atoi(argv[3]);

    cout << "My root is " << root << endl;

    omp_set_num_threads(numGPUs);

    butterfly_communication bfComm[numGPUs];
    eoff_t edgeSplits [numGPUs+1];

        #pragma omp parallel
        {      
            int64_t thread_id = omp_get_thread_num ();
            // int64_t num_threads = omp_get_num_threads();
            cudaSetDevice(thread_id);

            for(int64_t g=0; g<numGPUs; g++){
                if(g!=thread_id){
                    int isCapable;
                    cudaDeviceCanAccessPeer(&isCapable,thread_id,g);
                    if(isCapable==1){
                        cudaDeviceEnablePeerAccess(g,0);
                    }
                }

            }



            vid_t nV = graph.nV();
            eoff_t nE = graph.nE();

            vid_t upperNV = nV;
            if(upperNV%numGPUs){
                upperNV = nV - (nV%numGPUs) + numGPUs;
            }

            vid_t upperNE = nE;
            if(upperNE%numGPUs){
                upperNE = nE - (nE%numGPUs) + numGPUs;
            }


            // int64_t my_start = (thread_id  ) * upperNV / num_threads;
            // int64_t my_end   = (thread_id+1) * upperNV / num_threads;
            // if(my_end>nV){
            //     my_end=nV;
            // }

            // eoff_t edgeVal = ((thread_id) * upperNE) /num_threads ;
            // vid_t start = binarySearch(graph.csr_out_offsets(),0, nV, edgeVal);

            eoff_t edgeVal = ((thread_id+1) * upperNE) /numGPUs ;
            if (edgeVal>nE)
                edgeVal = nE;
            edgeSplits[thread_id+1] = vertexBinarySearch(graph.csr_out_offsets(),0, nV+1, edgeVal);

            
            if(thread_id == 0 )
                edgeSplits[0]=0;
            

/*
            vid_t myNE = graph.csr_out_offsets()[my_end]-graph.csr_out_offsets()[my_start];

            eoff_t  *offsets = (eoff_t*)malloc(sizeof(eoff_t)*nV+1);
            vid_t *edges     = (vid_t*)malloc(sizeof(vid_t)*myNE+1);

            memcpy(edges,graph.csr_out_edges()+graph.csr_out_offsets()[my_start],sizeof(vid_t)*myNE);
            memcpy(offsets,graph.csr_out_offsets(),sizeof(vid_t)*(nV+1));


            for(vid_t v=0; v<NV;v++){

            }

            std::cout << " " << thread_id << " " << my_start << " " << my_end << " " << myNE << std::endl;
*/
            HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                                   graph.csr_out_edges());
            HornetGraph hornet_graph(hornet_init);
        
            #pragma omp barrier

            int64_t my_start,my_end ;

            my_start = edgeSplits[thread_id];
            my_end  = edgeSplits[thread_id+1];

            // printf("%ld %ld %ld %d %d\n", thread_id,my_start,my_end,edgeVal, graph.csr_out_offsets()[my_end]-graph.csr_out_offsets()[my_start]);


            butterfly bfs(hornet_graph,fanout);
            bfs.reset();
            bfs.setInitValues(root, my_start, my_end,thread_id);
            #pragma omp barrier
            if(thread_id==0){
                // cudaProfilerStart();
                TM.start();                    
            }

            bfs.queueRoot();

            #pragma omp barrier


            int front = 1;
            degree_t countTraversed=1;
            while(true){

                bfs.oneIterationScan(front);
                bfComm[thread_id].queue_remote_ptr = bfs.remoteQueuePtr();

                bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();
                // if(thread_id==0){
                //     for(int t=0; t<numGPUs;t++){
                //         cout << bfComm[thread_id].queue_remote_length << " ";
                //     }
                //     cout << endl;
                // }
            // printf("HERE2\n");


                #pragma omp barrier


                if(fanout==1){
                    for (int l=0; l<logNumGPUs; l++){
                        bfs.communication(bfComm,numGPUs,l);
     
                        bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();
                        #pragma omp barrier


                    }
                }else if (fanout==4){

                    if(numGPUs==4){
                        bfs.communication(bfComm,numGPUs,0);
     
                        bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();
                        #pragma omp barrier                        
                    }
                    else if(numGPUs==16){

                        bfs.communication(bfComm,numGPUs,0);
                        bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();
                        #pragma omp barrier                        
     
                        bfs.communication(bfComm,numGPUs,1);
                        bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();
                        #pragma omp barrier                        


                    }else{
                        printf("Right not supporting fanout=4 for only 4 and 16 gpus\n");
                        exit(1);
                    }


                }
            // printf("HERE1\n");


    // /            #pragma omp barrier

                bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();

                bfs.oneIterationComplete();

                #pragma omp barrier

                bfComm[thread_id].queue_local_length = bfs.localQueueSize();

                #pragma omp barrier


                // if(thread_id==0){
                //     for(int t=0; t<numGPUs;t++){
                //         cout << "!!! " << t << " ";
                //         cout << bfComm[t].queue_local_length << " ";
                //         cout << bfComm[t].queue_remote_length << " ";
                //         cout << endl;
                //     }
                // }

                degree_t currFrontier=0;
                for(int t=0; t<numGPUs; t++){
                    currFrontier+=bfComm[t].queue_local_length;
                    countTraversed+=bfComm[t].queue_local_length;
                }


                front++;

                if(currFrontier==0){
                    if(thread_id==0){

                        TM.stop();
                        // cudaProfilerStop();
                        TM.print("Butterfly BFS");
                        std::cout << "Number of levels is : " << front << std::endl;

                        std::cout << "The number of traversed vertices is : " << countTraversed << std::endl;
                    }
                    break;
                }


            }

            // if(offsets!=NULL)
            //     free(offsets);
            // if(edges!=NULL)
            //     free(edges);
            // bc.run();

        }


}
