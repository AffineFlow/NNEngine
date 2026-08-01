#pragma once
#include <vector>

#include "autograd/Op.hpp"
#include "autograd/Tensor.hpp"

namespace affineflow::autograd::ops {

class MaxPool2dOp : public Op {
  Tensor *x_, *out_;
  int channels_, in_h_, in_w_, kernel_size_, stride_, pad_;
  int max_batch_ = 0;

  // Stores the flat indices of the maximum elements for O(1) gradient routing
  std::vector<int> max_indices_;

 public:
  MaxPool2dOp(Tensor* x, Tensor* out, int channels, int in_h, int in_w,
              int kernel_size, int stride, int pad);
  void forward() override;
  void backward() override;
};

class AvgPool2dOp : public Op {
  Tensor *x_, *out_;
  int channels_, in_h_, in_w_, kernel_size_, stride_, pad_;

 public:
  AvgPool2dOp(Tensor* x, Tensor* out, int channels, int in_h, int in_w,
              int kernel_size, int stride, int pad);
  void forward() override;
  void backward() override;
};

}  // namespace affineflow::autograd::ops