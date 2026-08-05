#pragma once
#include <vector>

#include "autograd/Op.hpp"
#include "autograd/Tensor.hpp"

namespace affineflow::nn::autograd::ops {

class Conv2dOp : public Op {
  Tensor *x_, *w_, *bias_, *out_;
  int in_channels_, out_channels_, in_h_, in_w_, kernel_size_, stride_, pad_;

  int max_batch_ = 0;
  int num_threads_ = 1;
  std::vector<affineflow::nn::FlatStorage> cols_;
  std::vector<affineflow::nn::FlatStorage> thread_dW_;
  std::vector<affineflow::nn::FlatStorage> thread_db_;
  std::vector<affineflow::nn::FlatStorage> thread_dcol_;

 public:
  Conv2dOp(Tensor* x, Tensor* w, Tensor* bias, Tensor* out, int in_channels,
           int out_channels, int in_h, int in_w, int kernel_size, int stride,
           int pad);
  void forward() override;
  void backward() override;
};

}  // namespace affineflow::nn::autograd::ops