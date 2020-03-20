
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off

#include <vector>

#include <omp.h>

#include "Static/butterfly/butterfly-bfs.cuh"
#include "Static/butterfly/butterfly-bfsOperators.cuh"

using namespace std;
#include <array>

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

template <typename t,typename pos_t>
pos_t vertexBinarySearch(const t *offsets, pos_t l, pos_t r, t x) 
{ 
    if (r >= l) { 
        pos_t mid = l + (r - l) / 2L; 
  
        // If the element is present at the middle itself 
        if (offsets[mid] == x) // perfect load balancing
            return mid; 
  
        // Check left subarray
        if (offsets[mid] > x) 
            return vertexBinarySearch(offsets, l, mid - 1L, x); 
        else
        // Check right subarray 
            return vertexBinarySearch(offsets, mid + 1L, r, x); 
    } 
  
    // Return the vertex id of the smallest vertex with an offset greater than x. 
    return l; 
} 


int main(int argc, char* argv[]) {

    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;
    using namespace graph;

    // GraphStd<vert_t, vert_t> graph(UNDIRECTED);
    graph::GraphStd<int64_t,int64_t> graph(UNDIRECTED );
    graph.read(argv[1]);
    // graph.read(argv[1], RANDOMIZE);
    // CommandLineParam cmd(graph, argc, argv,false);
    Timer<DEVICE> TM;

    // int numGPUs=2; int logNumGPUs=1;
     // int numGPUs=4; int logNumGPUs=2;
    // int numGPUs=8; int logNumGPUs=3;
    // int numGPUs=16; int logNumGPUs=4;


    int64_t numGPUs=4; int64_t logNumGPUs=2; int64_t fanout=1;
    int64_t minGPUs=1,maxGPUs=16;
    // bool isLrb=false;
    int isLrb=0,onlyLrb=0,onlyFanout4=0;

    vert_t startRoot = (vert_t)graph.max_out_degree_id();
    vert_t root = startRoot;

    if (argc>=3){
        minGPUs = atoi(argv[2]);
    }
    if (argc>=4){
        maxGPUs = atoi(argv[3]);
    }

    if (argc>=5){
        onlyLrb = atoi(argv[4]);
    }
    if (argc>=6){
        onlyFanout4 = atoi(argv[5]);
    }

    // omp_set_num_threads(numGPUs);

    omp_set_num_threads(maxGPUs);

    #pragma omp parallel
    {      
        int64_t thread_id = omp_get_thread_num ();
        cudaSetDevice(thread_id);

        for(int64_t g=0; g<maxGPUs; g++){
            if(g!=thread_id){
                int isCapable;
                cudaDeviceCanAccessPeer(&isCapable,thread_id,g);
                if(isCapable==1){
                    cudaDeviceEnablePeerAccess(g,0);
                }
            }
        }
    }

    cudaSetDevice(0);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    butterfly_communication bfComm[numGPUs];
    int64_t edgeSplits [16+1];
    int fanoutArray[2]={1,4};
    // std::array<int,2> fanoutArray{1,4};


    // for(int f=0; f<(int)fanoutArray.size() ; f++){
    for(int f=0; f<2 ; f++){
        if(f==0 && onlyFanout4)
            continue;
        fanout=fanoutArray[f];
    for(int lrb=0; lrb<2; lrb++){
        if(lrb==0 && onlyLrb)
            continue;
        isLrb=lrb;
    for(int g=minGPUs; g<=maxGPUs;g++){
        numGPUs=g;
        // int divg=numGPUs;

        int logNumGPUsArray[17] = {0,1,2,2,2,3,3,3,3,4,4,4,4,4,4,4,4};
        logNumGPUs = logNumGPUsArray[numGPUs];
        // logNumGPUs=0;
        // while(divg/2){
        //     logNumGPUs++;
        //     divg=divg/2;
        // }
        // if(divg>0 && numGPUs%)
        //     logNumGPUs++;

        // logNumGPUs = (int) log2(numGPUs); 
        // logNumGPUs = 31 -  __builtin_clz(numGPUs);
        omp_set_num_threads(numGPUs);


        root=startRoot;

        printf("%s,",argv[1]);
        printf("%ld,%ld,",graph.nV(),graph.nE());
        printf("%ld,",numGPUs);
        printf("%ld,",logNumGPUs);        
        printf("%ld,",fanout);
        printf("%d,",isLrb);
        printf("%d,",startRoot); // Starting root


            // printf("Waiting for enter\n");
            // int stam=0;
            // stam = scanf("%d",&stam);
            // printf("%d\n",stam+1);


        for(int64_t i=0; i<10; i++){

            if(i>0){
                root++;
                if(root>graph.nV())
                    root=0;
            }


            #pragma omp parallel
            {      
                int64_t thread_id = omp_get_thread_num ();
               // if(thread_id==0){
               //     printf(", %d ,",omp_get_num_threads() );
               // }
                cudaSetDevice(thread_id);

                int64_t nV = graph.nV(); int64_t nE = graph.nE();

                int64_t upperNV = nV;
                if(upperNV%numGPUs){
                    upperNV = nV - (nV%numGPUs) + numGPUs;
                }
                int64_t upperNE = nE;
                if(upperNE%numGPUs){
                    upperNE = nE - (nE%numGPUs) + numGPUs;
                }

                int64_t edgeVal = ((thread_id+1L) * upperNE) /numGPUs ;
                if (edgeVal>nE)
                    edgeVal = nE;
                int64_t zero=0;
                edgeSplits[thread_id+1] = vertexBinarySearch(graph.csr_out_offsets(),zero, nV+1L, (edgeVal));
                // printf("%ld %ld %ld\n",thread_id,edgeSplits[thread_id+1],edgeVal);

                if(thread_id == 0 )
                    edgeSplits[0]=0;

                #pragma omp barrier

                int64_t my_start,my_end,my_edges;

                my_start = edgeSplits[thread_id];
                my_end  = edgeSplits[thread_id+1];
                my_edges = graph.csr_out_offsets()[my_end]-graph.csr_out_offsets()[my_start];

                vert_t* localOffset = (vert_t*)malloc(sizeof(vert_t)*(nV+1));
                vert_t* edges       = (vert_t*)malloc(sizeof(vert_t)*(my_edges));

                int64_t i=0;
                for(int64_t u=my_start; u<my_end; u++){
                    int64_t d_size=graph.csr_out_offsets()[u+1]-graph.csr_out_offsets()[u];
                    for (int64_t d=0; d<d_size; d++){
                        edges[i++]=(vert_t) graph.csr_out_edges()[(graph.csr_out_offsets()[u]+d)];
                    }
                }

                // printf("%ld %ld %ld %ld %ld %ld\n", thread_id,my_start,my_end, my_edges,graph.csr_out_offsets()[my_start],graph.csr_out_offsets()[my_end]);
                fflush(stdout);

                for(int64_t v=0; v<(nV+1); v++){
                    localOffset[v]=0;
                }
                for(int64_t v=(my_start); v<nV; v++){
                    localOffset[v+1] = localOffset[v] + (graph.csr_out_offsets()[v+1]-graph.csr_out_offsets()[v]);
                }

                HornetInit hornet_init((vert_t)nV, (vert_t)my_edges, localOffset, edges);
                HornetGraph hornet_graph(hornet_init);

                butterfly bfs(hornet_graph,fanout);

                #pragma omp barrier
                if(thread_id==0){
                    // TM.start();   
                    cudaEventRecord(start); 
                    cudaEventSynchronize(start); 
                }

                bfs.reset();    
                bfs.setInitValues(root, my_start, my_end,thread_id);

                bfs.queueRoot();

                #pragma omp barrier

                int front = 1;
                degree_t countTraversed=1;
                while(true){
                    bfs.oneIterationScan(front,isLrb);
                    bfComm[thread_id].queue_remote_ptr = bfs.remoteQueuePtr();
                    bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();

                    #pragma omp barrier

                    if(fanout==1){
                        for (int l=0; l<logNumGPUs; l++){
                            bfs.communication(bfComm,numGPUs,l);
         
                            bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();
                            #pragma omp barrier
                        }
                    }else if (fanout==4){
                        // if(numGPUs==4){
                        //     bfs.communication(bfComm,numGPUs,0);
         
                        //     bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();
                        //     #pragma omp barrier                        
                        // }
                        // else{ //if(numGPUs==16){

                            bfs.communication(bfComm,numGPUs,0);
                            bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();
                            #pragma omp barrier                        
                            if(numGPUs>4){
                                bfs.communication(bfComm,numGPUs,1);
                                bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();
                                #pragma omp barrier                                                        
                            }
                        // }
                    }
        // /            #pragma omp barrier

                    bfComm[thread_id].queue_remote_length = bfs.remoteQueueSize();
                    bfs.oneIterationComplete();

                    #pragma omp barrier
                    bfComm[thread_id].queue_local_length = bfs.localQueueSize();

                    #pragma omp barrier

                    degree_t currFrontier=0;
                    for(int64_t t=0; t<numGPUs; t++){
                        currFrontier+=bfComm[t].queue_local_length;
                        countTraversed+=bfComm[t].queue_local_length;
                    }

                    front++;
                    if(currFrontier==0){
                        break;
                    }
           
                }
                #pragma omp barrier

                if(thread_id==0){

                    // TM.stop();
                    // cudaProfilerStop();
                    // TM.print("Butterfly BFS");
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    float milliseconds = 0;
                    cudaEventElapsedTime(&milliseconds, start, stop);  
                    printf("%f,", milliseconds/1000.0);             
                    // std::cout << "Number of levels is : " << front << std::endl;
                    // std::cout << "The number of traversed vertices is : " << countTraversed << std::endl;
                }

                if(localOffset!=NULL)
                    free(localOffset);
                if(edges!=NULL)
                    free(edges);
            }
        }
        printf("\n");
    }

    }
    }


    omp_set_num_threads(maxGPUs);

    #pragma omp parallel
    {      
        int64_t thread_id = omp_get_thread_num ();
        cudaSetDevice(thread_id);

        for(int64_t g=0; g<numGPUs; g++){
            if(g!=thread_id){
                int isCapable;
                cudaDeviceCanAccessPeer(&isCapable,thread_id,g);
                if(isCapable==1){
                    cudaDeviceDisablePeerAccess(g);
                }
            }
        }
    }

        return 0;
}
