#include "paddle/extension.h"

template <typename data_t>
__global__ void relu_cuda_forward_kernel(const data_t *x, data_t *y,
                                         const int num) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
    y[i] = max(x[i], static_cast<data_t>(0.));
  }
}

template <typename data_t>
__global__ void relu_cuda_backward_kernel(const data_t *dy, const data_t *y,
                                          data_t *dx, const int num) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
    dx[i] = dy[i] * (y[i] > 0 ? 1. : 0.);
  }
}

std::vector<paddle::Tensor> relu_cuda_forward(const paddle::Tensor &x) {
  auto out = paddle::empty_like(x);
  ;

  int numel = x.size();
  int block = 512;
  int grid = (numel + block - 1) / block;
  PD_DISPATCH_FLOATING_TYPES(
      x.type(), "relu_cuda_forward_kernel", ([&] {
        relu_cuda_forward_kernel<data_t><<<grid, block, 0, x.stream()>>>(
            x.data<data_t>(), out.data<data_t>(), numel);
      }));

  return {out};
}

std::vector<paddle::Tensor> relu_cuda_backward(const paddle::Tensor &x,
                                               const paddle::Tensor &out,
                                               const paddle::Tensor &grad_out) {
  auto grad_x = paddle::empty_like(x);
  ;

  int numel = out.size();
  int block = 512;
  int grid = (numel + block - 1) / block;
  PD_DISPATCH_FLOATING_TYPES(
      out.type(), "relu_cuda_backward_kernel", ([&] {
        relu_cuda_backward_kernel<data_t><<<grid, block, 0, x.stream()>>>(
            grad_out.data<data_t>(), out.data<data_t>(), grad_x.data<data_t>(),
            numel);
      }));

  return {grad_x};
}
