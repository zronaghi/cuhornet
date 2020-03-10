/**
 * @brief Breadth-first Search Top-Down test program
 * @file
 */
#include "Static/BFS-Reverse/BFS-Reverse.cuh"
#include <StandardAPI.hpp>
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off

int exec(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornets_nest;
    using vid_t = int;
    using dst_t = int;

    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;

    // graph::GraphStd<vid_t, eoff_t> graph;
    graph::GraphStd<vid_t, eoff_t> graph(DIRECTED | ENABLE_INGOING);
    CommandLineParam cmd(graph, argc, argv,false);


    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());


    HornetInit hornet_init_inverse(graph.nV(), graph.nE(),
                                   graph.csr_in_offsets(),
                                   graph.csr_in_edges());




    HornetGraph hornet_graph_inv(hornet_init_inverse);


        // std::cout << "hornet_graph_inv : " << hornet_graph_inv.nV() << " " << hornet_graph_inv.nE() << std::endl;
        // std::cout << "hornet_graph     : " << hornet_graph.nV() << " " << hornet_graph.nE() << std::endl;

    // for(int i=0; i<10; i++){
    //     std::cout << graph.csr_in_offsets()[i] << " : " << graph.csr_out_offsets()[i] << std::endl;
    // }

 
	vid_t root = graph.max_out_degree_id();
	// if (argc==3)
	//   root = atoi(argv[2]);

    int numberRoots = 10;
    if (argc>=3)
      numberRoots = atoi(argv[2]);

    int alg = 0;
    if (argc>=4)
      alg = atoi(argv[3]);

    int timeSection = 0;
    if (argc>=5)
      timeSection = atoi(argv[4]);


    std::cout << "My root is " << root << std::endl;

    // rev_del_bfs.set_parameters(root);

    float totalTime = 0.0;

    Timer<DEVICE> TM;

    for (int i=0; i<numberRoots;i++){
        HornetGraph hornet_graph(hornet_init);
        ReverseDeleteBFS rev_del_bfs(hornet_graph, hornet_graph_inv);
        rev_del_bfs.reset();

        // cudaProfilerStart();
        TM.start();
            rev_del_bfs.set_parameters((root+i)%graph.nV());
            rev_del_bfs.run(hornet_graph_inv,alg,timeSection);
        TM.stop();
        // printf("duration %f\n",TM.duration());
        totalTime += TM.duration();
        // cudaProfilerStop();
    }

    printf("\nReverse BFS time: %f ms\n",totalTime);

    // TM.print("Reverse BFS");

    return 0;
}

int main(int argc, char* argv[]) {
    int ret = 0;
    hornets_nest::gpu::initializeRMMPoolAllocation();//update initPoolSize if you know your memory requirement and memory availability in your system, if initial pool size is set to 0 (default value), RMM currently assigns half the device memory.
    {//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.

    ret = exec(argc, argv);

    }//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
    hornets_nest::gpu::finalizeRMMPoolAllocation();

    return ret;
}

