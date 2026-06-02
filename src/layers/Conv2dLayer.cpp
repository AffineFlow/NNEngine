#include "layers/Conv2dLayer.hpp"

#include <cmath>
#include <memory>

#include "autograd/Tape.hpp"
#include "autograd/ops/Conv2dOp.hpp"
#include "core/Random.hpp"

namespace mlengine::layers {

Conv2dLayer::Conv2dLayer(int in_channels, int out_channels, int in_h, int in_w,
                         int kernel_size, int stride, int pad)
    : w_(core::MatrixRM::Zero(out_channels,
                              in_channels * kernel_size * kernel_size),
         true),
      bias_(core::MatrixRM::Zero(1, out_channels), true),
      in_channels_(in_channels),
      out_channels_(out_channels),
      in_h_(in_h),
      in_w_(in_w),
      kernel_size_(kernel_size),
      stride_(stride),
      pad_(pad) {
  float stdv = std::sqrt(2.0f / (in_channels * kernel_size * kernel_size));
  float* w_ptr = w_.data.data();
  int size = w_.data.size();
  for (int i = 0; i < size; ++i) {
    w_ptr[i] = core::random_normal(0.0f, stdv);
  }
}

autograd::Tensor* Conv2dLayer::forward(autograd::Tensor* input) {
  auto* tape = autograd::Tape::get_global();
  bool req_grad =
      tape->record_ops_ &&
      (input->requires_grad || w_.requires_grad || bias_.requires_grad);

  int out_h = (in_h_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int out_w = (in_w_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  auto* out = tape->alloc_tensor(input->data.rows(),
                                 out_channels_ * out_h * out_w, req_grad);

  auto op = std::make_shared<autograd::ops::Conv2dOp>(
      input, &w_, &bias_, out, in_channels_, out_channels_, in_h_, in_w_,
      kernel_size_, stride_, pad_);
  op->forward();
  tape->record_op(op);
  return out;
}

std::vector<autograd::Tensor*> Conv2dLayer::parameters() {
  return {&w_, &bias_};
}

std::map<std::string, autograd::Tensor*> Conv2dLayer::named_parameters() {
  return {{"weight", &w_}, {"bias", &bias_}};
}

}  // namespace mlengine::layers