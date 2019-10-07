#pragma once

#include "HornetAlg.hpp"


namespace hornets_nest {

//using triangle_t = int;
using triangle_t = unsigned long long;
using vid_t = int;

using HornetGraph = ::hornet::gpu::Hornet<vid_t>;
using HornetInit  = ::hornet::HornetInit<vid_t>;


//==============================================================================

template<typename HornetClass>
//class SpGEMM : public StaticAlgorithm<HornetGraph> {
class SpGEMM {
public:
    SpGEMM(HornetClass& hornetA, HornetClass& hornetB);
    ~SpGEMM();

    void reset();
    void run();
    void release();
    bool validate(){ return true; }

    void run(const int WORK_FACTOR);
    void init();
    // void copyTCToHost(triangle_t* h_tcs);

    // triangle_t countTriangles();

protected:
   triangle_t* triPerVertex { nullptr };

private:
    HornetClass& hornetA;
    HornetClass& hornetB;

};

//==============================================================================

} // namespace hornets_nest
