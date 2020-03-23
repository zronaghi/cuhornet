#pragma once

#include "HornetAlg.hpp"


namespace hornets_nest {

//using triangle_t = int;
// using triangle_t = unsigned long long;
using triangle_t = int32_t;
using vid_t = int;

using HornetGraph = ::hornet::gpu::Hornet<vid_t>;
using HornetInit  = ::hornet::HornetInit<vid_t>;


//==============================================================================

// template<typename HornetGraph>
//class SpGEMM : public StaticAlgorithm<HornetGraph> {
class SpGEMM {
public:
    SpGEMM(HornetGraph& hornetA, HornetGraph& hornetB, HornetGraph& hornetC);
    ~SpGEMM();

    void reset();
    void run();
    void release();
    bool validate(){ return true; }

    void run(const int WORK_FACTOR);
    void init();
    
    vid_t rowc;
    vid_t colc;
    vid2_t* vertex_pairs;
    // void copyTCToHost(triangle_t* h_tcs);

    // triangle_t countTriangles();

protected:
   triangle_t* triPerVertex { nullptr };

private:
    HornetGraph& hornetA;
    HornetGraph& hornetB;
    HornetGraph& hornetC;

};

//==============================================================================

} // namespace hornets_nest
