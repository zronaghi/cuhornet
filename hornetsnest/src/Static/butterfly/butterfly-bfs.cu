


#include "Static/butterfly/butterfly-bfs.cuh"
#include "Static/butterfly/butterfly-bfsOperators.cuh"


using length_t = int;
using namespace std;
using namespace hornets_nest::gpu;
namespace hornets_nest {


struct countDegrees {
    int32_t *bins; 


    OPERATOR(Vertex& vertex) {

        __shared__ int32_t localBins[33];
        int id = threadIdx.x;
        if(id==0){
            for (int i=0; i<33; i++)
            localBins[i]=0;
        }
        __syncthreads();

        int32_t size = vertex.degree();
        int32_t myBin  = __clz(size);

        int32_t my_pos = atomicAdd(localBins+myBin, 1);

        __syncthreads();

       if(id==0){
            for (int i=0; i<33; i++){            
                atomicAdd(bins+i, localBins[i]);
            }

        }

    }
};





__global__ void  binPrefixKernel(int32_t     *bins, int32_t     *d_binsPrefix){

    int i = threadIdx.x + blockIdx.x *blockDim.x;
    if(i>=1)
        return;
    d_binsPrefix[0]=0;
    for(int b=0; b<33; b++){
        d_binsPrefix[b+1]=d_binsPrefix[b]+bins[b];
        // printf("%d ",d_binsPrefix[b+1] );  
    }
    // printf("\n");


}

template<typename HornetDevice>
__global__ void  rebinKernel(
  HornetDevice hornet ,
  const vert_t    *original,
  int32_t    *d_binsPrefix,
  vert_t     *d_reOrg,
  int N){

    int i = threadIdx.x + blockIdx.x *blockDim.x;

    __shared__ int32_t localBins[33];
    __shared__ int32_t localPos[33];

    __shared__ int32_t prefix[33];    
    int id = threadIdx.x;
    if(id<33){
      localBins[id]=0;
      localPos[id]=0;
    }

    __syncthreads();

    int myBin,myPos;
    if(i<N){
        int32_t adjSize= hornet.vertex(original[i]).degree();
        myBin  = __clz(adjSize);
        myPos = atomicAdd(localBins+myBin, 1);
    }


  __syncthreads();
    if(id<33){
        localPos[id]=atomicAdd(d_binsPrefix+id, localBins[id]);
    }
  __syncthreads();

    if(i<N){
        int pos = localPos[myBin]+myPos;
        d_reOrg[pos]=original[i];
    }

}









/// TODO - changed hostKatzdata to pointer so that I can try to inherit it in
// the streaming case.

butterfly::butterfly(HornetGraph& hornet, int fanout_) :
                                       StaticAlgorithm(hornet),
                                       load_balancing(hornet)
{

    fanout=fanout_;
    if(fanout!=4 && fanout!=1){
        printf("Fanout has to be 1 or 4 for butterfly `BFS\n");
        exit(0);
    }

    hd_bfsData().currLevel=0;

    gpu::allocate(hd_bfsData().d_buffer, fanout*hornet.nV());
    gpu::allocate(hd_bfsData().d_Marked, hornet.nV());
    gpu::allocate(hd_bfsData().d_dist, hornet.nV());

    gpu::allocate(hd_bfsData().d_lrbRelabled, hornet.nV());
    gpu::allocate(hd_bfsData().d_bins, 33);
    gpu::allocate(hd_bfsData().d_binsPrefix, 33);


    // gpu::allocate(hd_bfsData().queueRemote, hornet.nV());
    // hd_bfsData().queueRemoteSize=0;

    // hd_bfsData().queueLocal.initialize(hornet);
    // hd_bfsData().queueRemote.initialize(hornet);

    hd_bfsData().queueLocal.initialize((size_t)hornet.nV());
    hd_bfsData().queueRemote.initialize((size_t)hornet.nV());


    reset();
}

butterfly::~butterfly() {
    release();
}

void butterfly::setInitValues(vert_t root_ ,vert_t lower_, vert_t upper_,int64_t gpu_id_)
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
    gpu::free(hd_bfsData().d_lrbRelabled);
    gpu::free(hd_bfsData().d_bins);
    gpu::free(hd_bfsData().d_binsPrefix);


    // gpu::free(hd_bfsData().queueRemote);

}

void butterfly::queueRoot(){
    
    // std::cout << " " << hd_bfsData().gpu_id << " " << hd_bfsData().lower << " " <<  hd_bfsData().upper << " " <<  hd_bfsData().root << std::endl;


    if (hd_bfsData().root >= hd_bfsData().lower && hd_bfsData().root <hd_bfsData().upper){
        hd_bfsData().queueLocal.insert(hd_bfsData().root);                   // insert source in the frontier
    }
    gpu::memsetZero(hd_bfsData().d_dist + hd_bfsData().root);

}


void butterfly::oneIterationScan(degree_t level,bool lrb){

    hd_bfsData().currLevel = level;
    if (hd_bfsData().queueLocal.size() > 0) {
        if(!lrb){
            forAllEdges(hornet, hd_bfsData().queueLocal, BFSTopDown_One_Iter { hd_bfsData },load_balancing);
        }
        else{
            // hd_bfsData().queueLocal
            int32_t elements = hd_bfsData().queueLocal.size();

            gpu::memsetZero(hd_bfsData().d_bins, 33);            

            forAllVertices(hornet, hd_bfsData().queueLocal,countDegrees{hd_bfsData().d_bins});

            binPrefixKernel <<<1,32>>> (hd_bfsData().d_bins,hd_bfsData().d_binsPrefix);  

            int32_t h_binsPrefix[33];
            cudaMemcpy(h_binsPrefix, hd_bfsData().d_binsPrefix,sizeof(int32_t)*33, cudaMemcpyDeviceToHost);

            // for(int i=0; i<33; i++){
            //     printf("%d ",h_binsPrefix[i]);
            // }
            // printf("\n" );

            const int RB_BLOCK_SIZE = 256;
            int rebinblocks = (elements)/RB_BLOCK_SIZE + (((elements)%RB_BLOCK_SIZE)?1:0);

            if(rebinblocks){
              rebinKernel<<<rebinblocks,RB_BLOCK_SIZE>>>(hornet.device(),hd_bfsData().queueLocal.device_input_ptr(),
                hd_bfsData().d_binsPrefix, hd_bfsData().d_lrbRelabled,elements);
            }


            // if(rebinblocks>0)
            //     BFSTopDown_One_Iter_kernel<<<rebinblocks,RB_BLOCK_SIZE>>>(hornet.device_side(),
            //         hd_bfsData,elements,0);

            const int bi = 26;
            // printf("starting point is %d\n",h_binsPrefix[bi]);
            // cudaStream_t streams[2];
            //   cudaStreamCreate ( &(streams[0]));
            //   cudaStreamCreate ( &(streams[1]));


            rebinblocks = (h_binsPrefix[bi]);
            if(rebinblocks>0){
                // printf("fat is running %d \n",h_binsPrefix[bi]);
                BFSTopDown_One_Iter_kernel_fat<<<rebinblocks,RB_BLOCK_SIZE>>>(hornet.device(),hd_bfsData,h_binsPrefix[bi]);            }



            rebinblocks = (elements-h_binsPrefix[bi])/RB_BLOCK_SIZE + (((elements-h_binsPrefix[bi])%RB_BLOCK_SIZE)?1:0);
            if(rebinblocks>0)
                BFSTopDown_One_Iter_kernel<<<rebinblocks,RB_BLOCK_SIZE>>>(hornet.device(),
                    hd_bfsData,elements-h_binsPrefix[bi],h_binsPrefix[bi]);


        }

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

    if(fanout==1){

        int but_net[4][16] = {
                                {1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14},
                                {2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13},
                                {4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11},
                                {8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7}
        };  

        int my_gpu = hd_bfsData().gpu_id;
        int copy_gpu=but_net[iteration][my_gpu];
        if(copy_gpu>=numGPUs){
            copy_gpu=numGPUs-1;
        }

        hd_bfsData().h_bufferSize=bfComm[copy_gpu].queue_remote_length;
        cudaMemcpyPeerAsync(hd_bfsData().d_buffer, my_gpu, bfComm[copy_gpu].queue_remote_ptr,copy_gpu, hd_bfsData().h_bufferSize*sizeof(vert_t));
        
        if (hd_bfsData().h_bufferSize > 0){
            forAllVertices(hornet, hd_bfsData().d_buffer, hd_bfsData().h_bufferSize, NeighborUpdates { hd_bfsData });

        }

    }else if(fanout==4){
        int but_net_first[16][4]={{0,1,2,3},{0,1,2,3},{0,1,2,3},{0,1,2,3},
                                  {4,5,6,7},{4,5,6,7},{4,5,6,7},{4,5,6,7},
                                  {8,9,10,11},{8,9,10,11},{8,9,10,11},{8,9,10,11},
                                  {12,13,14,15},{12,13,14,15},{12,13,14,15},{12,13,14,15}};

        int but_net_second[16][4]={{0,4,8,12},{1,5,9,13},{2,6,10,14},{3,7,11,15},
                                  {0,4,8,12},{1,5,9,13},{2,6,10,14},{3,7,11,15},
                                  {0,4,8,12},{1,5,9,13},{2,6,10,14},{3,7,11,15},
                                  {0,4,8,12},{1,5,9,13},{2,6,10,14},{3,7,11,15}};

        int my_gpu = hd_bfsData().gpu_id;

        hd_bfsData().h_bufferSize=0;
        int pos=0;
        for(int s=0; s<4;s++){
            int copy_gpu;
            if(iteration==0)
                copy_gpu=but_net_first[my_gpu][s];
            else
                copy_gpu=but_net_second[my_gpu][s];

            if(copy_gpu>=numGPUs){
                copy_gpu=numGPUs-1;
            }

            
            if(my_gpu!=copy_gpu){
                int remoteLength = bfComm[copy_gpu].queue_remote_length;                
                cudaMemcpyPeerAsync(hd_bfsData().d_buffer+pos, my_gpu, bfComm[copy_gpu].queue_remote_ptr,copy_gpu, remoteLength*sizeof(vert_t));
                pos+=remoteLength;
                hd_bfsData().h_bufferSize+=remoteLength;

            }
        }
        
        if (hd_bfsData().h_bufferSize > 0){
            forAllVertices(hornet, hd_bfsData().d_buffer, hd_bfsData().h_bufferSize, NeighborUpdates { hd_bfsData });

        }


    }


}



void butterfly::run() {



}


// int butterfly::getDepth() {
//     return hd_bfsData();
// }

bool butterfly::validate() {
    return true;
}

} // namespace hornets_nest
