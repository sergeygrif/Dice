// policy_gather.cu
#include <cuda_runtime.h>
#include <math_constants.h>

#ifndef POLICY_SIZE
#define POLICY_SIZE (73 * 64)   // 4672
#endif

#ifndef AI_MAX_MOVES
#define AI_MAX_MOVES 255
#endif

__global__ void gatherPolicyKernelBatched(const float* __restrict__ policy,
                                         const int*   __restrict__ idx,
                                         float*       __restrict__ out,
                                         int total)
{
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= total) return;

    // idx/out layout: [B * AI_MAX_MOVES]
    const int b = tid / AI_MAX_MOVES;
    const int k = idx[tid];

    // invalid / padded index => -inf, so softmax gives zero probability
    if ((unsigned)k >= (unsigned)POLICY_SIZE) {
        out[tid] = -CUDART_INF_F;
        return;
    }

    out[tid] = policy[b * POLICY_SIZE + k];
}

extern "C" void launchGatherPolicyKernel(const float* policy,
                                         const int* idx,
                                         float* out,
                                         int total,
                                         cudaStream_t stream)
{
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    gatherPolicyKernelBatched<<<blocks, threads, 0, stream>>>(policy, idx, out, total);
}
