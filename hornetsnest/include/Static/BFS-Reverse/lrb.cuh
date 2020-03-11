
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
    }
}


template<typename HornetDevice>
__global__ void  rebinKernel(
  HornetDevice hornet ,
  const vid_t    *original,
  int32_t    *d_binsPrefix,
  vid_t     *d_reOrg,
  int N){

    int i = threadIdx.x + blockIdx.x *blockDim.x;

    __shared__ int32_t localBins[33];
    __shared__ int32_t localPos[33];

    // __shared__ int32_t prefix[33];    
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






template<bool onlyNonDeleted, typename HornetDevice>
__global__ void BFSTopDown_One_Iter_kernel(
  HornetDevice hornet , 
  HostDeviceVar<invBFSData> bfs, 
  found_t*       d_found,
  TwoLevelQueue<vid_t> queue,
  int N,
  int start){
    int k = threadIdx.x + blockIdx.x *blockDim.x;
    if(k>=N)
        return;
    k+=start;

    vid_t src = bfs().d_lrbRelabled[k];
    degree_t currLevel = bfs().currLevel;

    vid_t* neighPtr = hornet.vertex(src).neighbor_ptr();
    int length = hornet.vertex(src).degree();

    for (int i=0; i<length; i++) {
       if(onlyNonDeleted){
            if(hornet.vertex(src).edge(i).template field<0>()==1)
                continue;
       }
       vid_t dst_id = neighPtr[i]; 

        if(d_found[dst_id] == 0){
            if (atomicCAS(d_found + dst_id, 0, 1) == 0) {
                queue.insert(dst_id);
            }
        }
    }
}

template<bool onlyNonDeleted, typename HornetDevice>
__global__ void BFSTopDown_One_Iter_kernel_fat(
  HornetDevice hornet , 
  HostDeviceVar<invBFSData> bfs,
  found_t*       d_found,
  TwoLevelQueue<vid_t> queue,
  int N,
  int start){
    int k = blockIdx.x;
    int tid = threadIdx.x;
    if(k>=N){
        printf("should never happen\n");
        return;
    }
    k+=start;    

    vid_t src = bfs().d_lrbRelabled[k];
    degree_t currLevel = bfs().currLevel;

    vid_t* neighPtr = hornet.vertex(src).neighbor_ptr();
    int length = hornet.vertex(src).degree();

    for (int i=tid; i<length; i+=blockDim.x) {

       if(onlyNonDeleted){
            if(hornet.vertex(src).edge(i).template field<0>()==1)
                continue;
       }

       vid_t dst_id = neighPtr[i]; 

        if(d_found[dst_id] == 0){
            if (atomicCAS(d_found + dst_id, 0, 1) == 0) {
                queue.insert(dst_id);
            }
        }
    }
}




} // namespace hornets_nest
