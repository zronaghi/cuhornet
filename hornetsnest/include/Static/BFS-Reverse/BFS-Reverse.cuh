
#pragma once


#include "HornetAlg.hpp"
#include <Graph/GraphStd.hpp>


#include <StandardAPI.hpp>
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off
using namespace timer;


namespace hornets_nest {


using vid_t = int;
using HornetGraph = ::hornet::gpu::Hornet<vid_t>;
using HornetInit  = ::hornet::HornetInit<vid_t>;


using UpdatePtr   = ::hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::DEVICE>;
using Update      = ::hornet::gpu::BatchUpdate<vert_t>;


using found_t = int;
using dist_t = int;
using batch_t = int;

class ReverseDeleteBFS : public StaticAlgorithm<HornetGraph> {
public:
    ReverseDeleteBFS(HornetGraph& hornet, HornetGraph& hornet_in);
    ~ReverseDeleteBFS();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override;
    void run(HornetGraph& hornet_in);

    void set_parameters(vid_t source);
private:
    TwoLevelQueue<vid_t>        queue;
    TwoLevelQueue<vid_t>        queue_inf;
    load_balancing::BinarySearch load_balancing;

    // dist_t* d_distances   { nullptr };
    found_t* d_found   { nullptr };
    vert_t* d_src { nullptr };
    vert_t* d_dest { nullptr };
    batch_t* d_batchSize { nullptr };


    vid_t   root    { 0 };
    dist_t  current_level { 0 };
};

const dist_t INF = std::numeric_limits<dist_t>::max();



//------------------------------------------------------------------------------
///////////////
// OPERATORS //
///////////////


struct findNew {
    found_t*       d_found;
    TwoLevelQueue<vid_t> queue;

    OPERATOR(Vertex& vertex, Edge& edge) {
        auto dst = edge.dst_id();
        if (atomicCAS(d_found + dst, 0, 1) == 0) {
            queue.insert(dst);
        }
    }
};

struct createBatch {
    found_t*  d_found;
    vert_t*   d_src;
    vert_t*   d_dest;
    batch_t*  d_batchSize;

    OPERATOR(Vertex& vertex, Edge& edge) {
        auto pos = atomicAdd(d_batchSize,1);

        // Notice that we are replacing the roles of src<-->dest as the batch is applied
        // to the second graph
        d_src[pos]  = edge.dst_id();
        d_dest[pos] = vertex.id();
    }
};


//------------------------------------------------------------------------------
//////////////////////
// ReverseDeleteBFS //
//////////////////////

ReverseDeleteBFS::ReverseDeleteBFS(HornetGraph& hornet, HornetGraph& hornet_inv) :
                                 StaticAlgorithm(hornet),
                                 queue(hornet),
                                 load_balancing(hornet) {
    gpu::allocate(d_found, hornet.nV());

    auto edges = hornet.nE();
    gpu::allocate(d_src, edges);
    gpu::allocate(d_dest, edges);
    gpu::allocate(d_batchSize, 1);


    reset();
}

void ReverseDeleteBFS::release() {
    if(d_found == nullptr){
        gpu::free(d_found);d_found = nullptr;
    }
    if(d_src == nullptr){
        gpu::free(d_src);d_src = nullptr;
    }
    if(d_dest == nullptr){
        gpu::free(d_dest);d_dest = nullptr;
    }
    if(d_batchSize == nullptr){
        gpu::free(d_batchSize);d_batchSize = nullptr;
    }

}

ReverseDeleteBFS::~ReverseDeleteBFS() {
    release();
}

void ReverseDeleteBFS::reset() {
    current_level = 1;
    queue.clear();
    auto found = d_found;

    forAllnumV(hornet, [=] __device__ (int i){ found[i] = 0; } );
    found_t rootfound=1;
    cudaMemcpy(d_found+root,&rootfound,sizeof(found_t),cudaMemcpyHostToDevice);
}

void ReverseDeleteBFS::set_parameters(vid_t root_) {
    root = root_;
    queue.insert(root);// insert bfs source in the frontier
    // gpu::memsetZero(d_distances + root);  //reset source distance
}

void ReverseDeleteBFS::run() {

    printf("This function is not doing anything right now. Function requires overriding\n");
}

void ReverseDeleteBFS::run(HornetGraph& hornet_inv) {

    Timer<DEVICE> TM;
    // cudaProfilerStart();
    TM.start();

    int level=1;
    float traversal = 0, batchtime = 0;
    int total_counter = 0;
    while (queue.size() > 0) {

        // TM.start();
        gpu::memsetZero(d_batchSize);  //reset source distance

        forAllEdges(hornet_inv, queue,
            createBatch { d_found, d_src, d_dest, d_batchSize},load_balancing);
        // TM.stop();
        // traversal += TM.duration();
        // TM.print("Batch creation");

        // TM.start();

        batch_t h_counter;
        cudaMemcpy(&h_counter,d_batchSize, sizeof(batch_t),cudaMemcpyDeviceToHost);
        // printf("h_counter = %d\n", h_counter);total_counter+=h_counter;
        UpdatePtr ptr(h_counter, d_src, d_dest);
        Update batch_update(ptr);
        hornet.erase(batch_update);

        // TM.stop();
        // batchtime += TM.duration();
        // TM.print("Batch deletion");
        // TM.start();

        TM.start();
        if(1){
            forAllEdges(hornet, queue,
                    findNew { d_found, queue},load_balancing);
        }else{

        }

        TM.stop();
        traversal += TM.duration();

        queue.swap();
        level++;

        // TM.stop();

    }

    printf("%f,",traversal);
    // printf("%f,",batchtime);

    // printf("Traversal time %f\n",traversal);
    // printf("Batch time     %f\n",batchtime);
    // printf("Number of deleted edges %d\n",total_counter);

    // TM.stop();
    // cudaProfilerStop();
    // TM.print("Reverse BFS");

    // printf("Number of levels is %d\n", level);
}



bool ReverseDeleteBFS::validate() {
  return true;
}

} // namespace hornets_nest

