


#include "Static/butterfly/butterfly-bfs.cuh"
#include "Static/butterfly/butterfly-bfsOperators.cuh"


using length_t = int;
using namespace std;
namespace hornets_nest {

/// TODO - changed hostKatzdata to pointer so that I can try to inherit it in
// the streaming case.

butterfly::butterfly(HornetGraph& hornet) :
                                       StaticAlgorithm(hornet),
                                       load_balancing(hornet)
{
    hd_bfsData().currLevel=0;

    gpu::allocate(hd_bfsData().d_buffer, hornet.nV());
    gpu::allocate(hd_bfsData().d_Marked, hornet.nV());
    gpu::allocate(hd_bfsData().d_dist, hornet.nV());

    // gpu::allocate(hd_bfsData().queueRemote, hornet.nV());
    // hd_bfsData().queueRemoteSize=0;

    hd_bfsData().queueLocal.initialize(hornet);
    hd_bfsData().queueRemote.initialize(hornet);


    reset();
}

butterfly::~butterfly() {
    release();
}

void butterfly::setInitValues(vid_t root_ ,vid_t lower_, vid_t upper_,int64_t gpu_id_)
{
    // if(gpu_id_==0)
    //     std::cout << " " << gpu_id_ << " " << lower_ << " " <<  upper_ << " " <<  root_ << std::endl;
    // hd_bfsData.sync();

    hd_bfsData().currLevel  = 1;
    hd_bfsData().root       = root_;
    hd_bfsData().lower      = lower_;
    hd_bfsData().upper      = upper_;
    hd_bfsData().gpu_id     = gpu_id_;
// /    hd_bfsData().queueRemoteSize=0;

    // if(gpu_id_==0)
    //     std::cout << " " << hd_bfsData().gpu_id << " " << hd_bfsData().lower << " " <<  hd_bfsData().upper << " " <<  hd_bfsData().root << std::endl;

    // if(gpu_id_==0)
    //     std::cout << " " << hd_bfsData().gpu_id << " " << hd_bfsData().lower << " " <<  hd_bfsData().upper << " " <<  hd_bfsData().root << std::endl;

}


void butterfly::reset() {

    forAllnumV(hornet, InitBFS { hd_bfsData });
    cudaDeviceSynchronize();

    // hd_bfsData.sync();
}

void butterfly::release(){
    gpu::free(hd_bfsData().d_buffer);
    gpu::free(hd_bfsData().d_Marked);
    gpu::free(hd_bfsData().d_dist);
    // gpu::free(hd_bfsData().queueRemote);

}

void butterfly::queueRoot(){
    
    // std::cout << " " << hd_bfsData().gpu_id << " " << hd_bfsData().lower << " " <<  hd_bfsData().upper << " " <<  hd_bfsData().root << std::endl;


    if (hd_bfsData().root >= hd_bfsData().lower && hd_bfsData().root <hd_bfsData().upper){
        hd_bfsData().queueLocal.insert(hd_bfsData().root);                   // insert source in the frontier
    }
    gpu::memsetZero(hd_bfsData().d_dist + hd_bfsData().root);

}


void butterfly::oneIterationScan(degree_t level){

    hd_bfsData().currLevel = level;
    if (hd_bfsData().queueLocal.size() > 0) {
        forAllEdges(hornet, hd_bfsData().queueLocal, BFSTopDown_One_Iter { hd_bfsData },load_balancing);

        // hd_bfsData.sync();

        // std::cout << hd_bfsData().gpu_id << " " << hd_bfsData().queueLocal.size_sync_out() << std::endl;
        // std::cout << hd_bfsData().gpu_id << " " << hd_bfsData().queueRemote.size_sync_out() << std::endl;
        // hd_bfsData().queueLocal.swap();
        // std::cout << hd_bfsData().gpu_id << " " << hd_bfsData().queueLocal.size() << std::endl;

    }

}

void butterfly::oneIterationComplete(){

    hd_bfsData().queueLocal.swap();

    hd_bfsData().queueRemote.clear();
    cudaDeviceSynchronize();

}


void butterfly::communication(butterfly_communication* bfComm, int numGPUs, int iteration){

    int but_net[4][16] = {
                            {1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14},
                            {2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13},
                            {4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11},
                            {8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7}
    };  

    int my_gpu = hd_bfsData().gpu_id;
    int copy_gpu=but_net[iteration][my_gpu];
    hd_bfsData().h_bufferSize=bfComm[copy_gpu].queue_remote_length;
    cudaMemcpyPeerAsync(hd_bfsData().d_buffer, my_gpu, bfComm[copy_gpu].queue_remote_ptr,copy_gpu, hd_bfsData().h_bufferSize*sizeof(vid_t));

   //  if(hd_bfsData().gpu_id==0){
   //       hd_bfsData().h_bufferSize=bfComm[1].queue_remote_length;
   //      cudaMemcpyPeer(hd_bfsData().d_buffer, 0, bfComm[1].queue_remote_ptr,1,hd_bfsData().h_bufferSize*sizeof(vid_t));
   //  }
   //  else{
   //      hd_bfsData().h_bufferSize=bfComm[0].queue_remote_length;
   //      cudaMemcpyPeer(hd_bfsData().d_buffer, 1, bfComm[0].queue_remote_ptr,0,hd_bfsData().h_bufferSize*sizeof(vid_t));
   // }
    // cudaDeviceSynchronize();
    
    if (hd_bfsData().h_bufferSize > 0){
        forAllVertices(hornet, hd_bfsData().d_buffer, hd_bfsData().h_bufferSize, NeighborUpdates { hd_bfsData });


        // cout << "Thread ID " << hd_bfsData().gpu_id  << "  Buffer size " << hd_bfsData().h_bufferSize << endl;
        // cout << "Thread ID " << hd_bfsData().gpu_id  << "  Local Queue size " << hd_bfsData().queueLocal.size_sync_out() << endl;
    }

}



void butterfly::run() {


    // // Initialization
    // hd_bfsData().currLevel=0;
    // forAllnumV(hornet, InitOneTree { hd_bfsData });
    // vid_t root = hd_bfsData().root;

    // hd_bfsData().queue.insert(root);                   // insert source in the frontier
    // gpu::memsetZero(hd_bfsData().d + root);

    // // Regular BFS
    // hd_bfsData().depth_indices[0]=1;

    // while (hd_bfsData().queue.size() > 0) {

    //     hd_bfsData().depth_indices[hd_bfsData().currLevel+1]=
    //     hd_bfsData().depth_indices[hd_bfsData().currLevel]+hd_bfsData().queue.size();

    //     forAllEdges(hornet, hd_bfsData().queue, BC_BFSTopDown { hd_bfsData }, load_balancing);
    //     // hd_bfsData.sync();
    //     hd_bfsData().currLevel++;
    //     hd_bfsData().queue.swap();
    // }


}


// int butterfly::getDepth() {
//     return hd_bfsData();
// }

bool butterfly::validate() {
    return true;
}

} // namespace hornets_nest
