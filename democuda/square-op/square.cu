
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


#define THREADS_PER_BLOCK 512

__global__ void square_cuda_forward_kernel(
    float* inp, float* out, int N
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
    {
        out[idx] = inp[idx] * inp[idx];
    }
}

torch::Tensor square_cuda_forward(const torch::Tensor& input)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");


    auto output = torch::empty_like(input);
    int n = input.numel(); // 获得所有元素个数

    const int threads = THREADS_PER_BLOCK;
    const int blocks = (n + threads - 1) / threads;

    square_cuda_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}


__global__ void square_backward_kernel(const float* grad_output, const float* input,
                                       float* grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = 2.0f * input[idx] * grad_output[idx];
    }
}

torch::Tensor square_cuda_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input
)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(grad_output.dtype() == torch::kFloat32, "grad_output must be float32");

    
    int n = input.numel(); // 获得所有元素个数
    
    auto grad_input = torch::empty_like(input);

    const int threads = THREADS_PER_BLOCK;
    const int blocks = (n + threads - 1) / threads;

    square_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        n
    );

    return grad_input;
}


