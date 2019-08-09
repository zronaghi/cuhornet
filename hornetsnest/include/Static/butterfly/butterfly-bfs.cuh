
#pragma once

#include "HornetAlg.hpp"


namespace hornets_nest {

using HornetGraph = gpu::Hornet<EMPTY, EMPTY>;

struct butterflyData {
    degree_t currLevel;
    vid_t       root;
    vid_t       lower;
    vid_t       upper;
    int64_t     gpu_id;


    TwoLevelQueue<vid_t> queueLocal;
    TwoLevelQueue<vid_t> queueRemote;
    // vid_t* queueRemote;
    // degree_t queueRemoteSize;


    vid_t*      d_buffer;
    vid_t       h_bufferSize;

    bool*       d_Marked;
    vid_t*       d_dist;
};


struct butterfly_communication{
    const vid_t*      queue_remote_ptr;
    degree_t          queue_remote_length;
    degree_t          queue_local_length;
};


class butterfly : public StaticAlgorithm<HornetGraph> {
public:
    butterfly(HornetGraph& hornet);
    ~butterfly();

    void setInitValues(vid_t root_,vid_t lower_, vid_t upper_,int64_t gpu_id);
    void queueRoot();
    

    // vid_t* remoteQueuePtr(){return hd_bfsData().queueRemote;}
    // vid_t remoteQueueSize(){return hd_bfsData().queueRemoteSize;}

    const vid_t*    remoteQueuePtr(){return hd_bfsData().queueRemote.device_output_ptr();}
    vid_t           remoteQueueSize(){return hd_bfsData().queueRemote.size_sync_out();}

    vid_t           localQueueSize(){return hd_bfsData().queueLocal.size_sync_in();}

    void oneIterationScan(degree_t level);

    void oneIterationComplete();
    void communication(butterfly_communication* bfComm, int numGPUs,int iteration);

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override;


// private:
    load_balancing::BinarySearch load_balancing;

    HostDeviceVar<butterflyData>       hd_bfsData;    

};

} // hornetAlgs namespace
