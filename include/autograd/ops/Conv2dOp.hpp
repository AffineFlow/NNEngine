#pragma once
#include <vector>

#include "autograd/Op.hpp"
#include "autograd/Tensor.hpp"

namespace mlengine::autograd::ops {

/**
 * @brief Native 2D Convolution operation using im2col algorithm.
 */
class Conv2dOp : public Op {
  Tensor *x_, *w_, *bias_, *out_;
  int in_channels_, out_channels_, in_h_, in_w_, kernel_size_, stride_, pad_;

  // Pre-allocated memory pools to prevent heap fragmentation during JIT replay
  int max_batch_ = 0;
  int num_threads_ = 1;
  std::vector<mlengine::MatrixRM> cols_;
  std::vector<mlengine::MatrixRM> thread_dW_;
  std::vector<mlengine::MatrixRM> thread_db_;
  std::vector<mlengine::MatrixRM> thread_dcol_;

 public:
  Conv2dOp(Tensor* x, Tensor* w, Tensor* bias, Tensor* out, int in_channels,
           int out_channels, int in_h, int in_w, int kernel_size, int stride,
           int pad);
  void forward() override;
  void backward() override;
};

}  // namespace mlengine::autograd::ops