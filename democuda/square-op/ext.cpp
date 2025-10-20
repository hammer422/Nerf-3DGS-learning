
#include <torch/extension.h>

torch::Tensor square_cuda_forward(const torch::Tensor& input);
torch::Tensor square_cuda_backward(const torch::Tensor& grad_output, const torch::Tensor& input);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("square_forward", &square_cuda_forward, "Square forward (CUDA)");
    m.def("square_backward", &square_cuda_backward, "Square backward (CUDA)");
}