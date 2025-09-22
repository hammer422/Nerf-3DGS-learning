

#define OFFSET(row, col, ld) ((row) * (ld) + (col))



__global__ void reduce_sum(
    float* inp, float* out, int N
)
{
    extern __shared__ float smem[]; // blockDim.x
    const int tid = threadIdx.x;
    int i = tid;
    float sum = 0;
    while(i < N)
    {
        sum += inp[i];
        i += blockDim.x * gridDim.x;
    }
    smem[i] = sum;
    __syncthreads();

    for(int offset=blockDim.x>>1; offset>=32; offset>>=1)
    {
        if(tid < offset)
        {
            smem[tid] += smem[tid + offset];
        }
        __syncthreads();
    }

    float sum2 = smem[i];
    for(int offset=16; offset>0; offset>>=1)
    {
        sum2 += __shfl_down_sync(0xFFFFFFFF, sum2, offset);
    }

    if(tid == 0)
        atomicAdd(output, sum2);

}


template<
    const int TILE_DIM
>
__global__ void transpose(
    float* inp, float* out,
    int M, int N
)
{
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int by = blockIdx.y;
    const int bx = blockIdx.x;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    __shared__ float smem[TILE_DIM][TILE_DIM+1]; // Bank conflict

    if(row < M && col < N)
    {
        smem[ty][tx] = inp[OFFSET(row, col, N)];
    }
    __syncthreads();


    row = bx * blockDim.x + ty;
    col = by * blockDim.y + tx;

    if(row < M && col < N)
    {
        out[OFFSET(row, col, M)] = smem[tx][ty];
    }

}


__global__ void softmax(
    float* inp, float* out,
    int N, int C
)
{
    // only 32 threads per block
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    float max = -INFINITY;
    int i = tid;
    while(i < C)
    {
        max = max(max, inp[OFFSET(bid, i, C)]);
        i += blockDim.x;
    }

    // reduce
    for(int offset=16;offset>0;offset>>=1)
    {
        max = max(max, __shfl_down_sync(0XFFFFFFFF, max, offset));
    }

    float max2 = __shfl_sync(0xFFFFFFFF, max, 0);

    float sum = 0;
    i = tid;
    while(i < C)
    {
        float tmp = exp(inp[OFFSET(bid, i, C)] - max);
        sum += tmp;
        out[OFFSET(bid, i, C)] = tmp;
        i += blockDim.x;
    }


    // reduce
    for(int offset=16;offset>0;offset>>=1)
    {
        sum += __shfl_down_sync(0XFFFFFFFF, sum, offset);
    }

    float sum2 = __shfl_sync(0xFFFFFFFF, sum, 0);


    i = tid;
    while(i < C)
    {
        out[OFFSET(bid, i, C)] /= sum2;
        i += blockDim.x;
    }

}


template<
    const int BM,
    const int BN,
    const int BK,
    const int TM,
    const int TN
>
__global__ void sgemm(
    float* A, float* B, float* C,
    int M, int N, int K
)
{
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int by = blockIdx.y;
    const int bx = blockIdx.x;

    __shared__ float smemA[BM][BK];
    __shared__ float smemB[BK][BN];

    A = A + OFFSET(by * BM, 0, K);
    B = B + OFFSET(0, bx * BN, N);
    C = C + OFFSET(by * BM, bx * BN, N);

    int row = ty * TM;
    int col = tx * TN;

    float tmp[TM][TN] = {0};
    for(int bk=0;bk<K;bk+=BK)
    {
        for(int i=0;i<TM;i++)
        {
            for(int k=0;k<BK;k++)
            {
                smemA[row+i][k] = A[OFFSET(row+i,bk+k,K)];
            }
        }
        for(int i=0;i<TN;i++)
        {
            for(int k=0;k<BK;k++)
            {
                smemB[k][col+i] = B[OFFSET(bk+k,col+i,N)];
            }
        }

        __syncthreads();

        for(int k=0;k<BK;k++)
        {
            for(int i=0;i<TM;i++)
            {
                for(int i=0;i<TN;i++)
                {
                    tmp[i][j] += smemA[row+i][k] * smemB[k][col+j];
                }
            }
        }


        __syncthreads();
    }


    // store

    for(int i=0;i<TM;i++)
    {
        for(int j=0;j<TN;j++)
        {
            C[OFFSET(row+i, col+j, N)] = tmp[i][j];
        }
    }


}