#pragma once


#include "HornetAlg.hpp"


namespace hornets_nest {

//using triangle_t = int;
// using triangle_t = unsigned long long;
// using triangle_t = int;
// using vid_t = int;

// using HornetGraph = ::hornet::gpu::Hornet<vid_t>;
// using HornetInit  = ::hornet::HornetInit<vid_t>;
// using UpdatePtr   = ::hornet::BatchUpdatePtr<vid_t, hornet::EMPTY, hornet::DeviceType::DEVICE>;
// using Update      = ::hornet::gpu::BatchUpdate<vid_t>;

using vid_t = int;
using wgt0_t = vid_t;
using triangle_t = wgt0_t;
using HornetInit   = ::hornet::HornetInit<vid_t, hornet::EMPTY, hornet::TypeList<wgt0_t>>;
using HornetGraph  = hornet::gpu::Hornet<vid_t, hornet::EMPTY, hornet::TypeList<wgt0_t>>;
using UpdatePtr    = hornet::BatchUpdatePtr<vid_t, hornet::TypeList<wgt0_t>, hornet::DeviceType::DEVICE>;
using Update       = hornet::gpu::BatchUpdate<vid_t, hornet::TypeList<wgt0_t>>;
using HornetGraphPtr = HornetGraph*;

//==============================================================================

// template<typename HornetGraph>
//class SpGEMM : public StaticAlgorithm<HornetGraph> {
class SpGEMM {
public:
    SpGEMM(HornetGraph* hornetA, HornetGraph* hornetB, HornetGraph* hornetC, 
           int concurrentIntersections, float workFactor, bool sanityCheck);
    ~SpGEMM();

    SpGEMM(HornetGraphPtr* hornetAarray, HornetGraphPtr* hornetBarray, HornetGraphPtr* hornetCarray,
           int numGPUs, int concurrentIntersections, float workFactor, bool sanityCheck);
    
    void reset();
    void run();
    void release();
    bool validate(){ return true; }

    void run(const int WORK_FACTOR);
    void init();
    
    vid_t rowc;
    vid_t colc;
    // vid2_t* vertex_pairs;
    // void copyTCToHost(triangle_t* h_tcs);

    // triangle_t countTriangles();

protected:
   triangle_t* triPerVertex { nullptr };

private:
    HornetGraph* hornetA;
    HornetGraph* hornetB;
    HornetGraph* hornetC;

    HornetGraphPtr* hornetAarray;
    HornetGraphPtr* hornetBarray;
    HornetGraphPtr* hornetCarray;

    int concurrentIntersections;
    float workFactor;
    bool sanityCheck;
    int numGPUs;
};


struct arrayExtraInfo{
public:
    arrayExtraInfo(degree_t size){
        hornets_nest::gpu::allocate(d_First, size);
        hornets_nest::gpu::allocate(d_Last, size);
        hornets_nest::gpu::allocate(d_Size, size);
        hornets_nest::gpu::allocate(d_Log, size);
    }

    ~arrayExtraInfo(){
        hornets_nest::gpu::free(d_First);
        hornets_nest::gpu::free(d_Last);
        hornets_nest::gpu::free(d_Size);
        hornets_nest::gpu::free(d_Log);
    }

    vid_t *d_First;
    vid_t *d_Last; 
    degree_t *d_Size;
    unsigned int *d_Log; 
};

//==============================================================================

} // namespace hornets_nest
