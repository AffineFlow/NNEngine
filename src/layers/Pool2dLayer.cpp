#include "layers/Pool2dLayer.hpp"

#include "autograd/Tape.hpp"
#include "autograd/ops/Pool2dOp.hpp"

namespace affineflow::layers {

MaxPool2dLayer::MaxPool2dLayer(int channels, int in_h, int in_w,
                               int kernel_size, int stride, int pad)
    : channels_(channels),
      in_h_(in_h),
      in_w_(in_w),
      kernel_size_(kernel_size),
      stride_(stride),
      pad_(pad) {}

autograd::Tensor* MaxPool2dLayer::forward(autograd::Tensor* input) {
  auto* tape = autograd::Tape::get_global();
  bool req_grad = tape->record_ops_ && input->requires_grad;

  int out_h = (in_h_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int out_w = (in_w_ + 2 * pad_ - kernel_size_) / stride_ + 1;

  auto* out = tape->alloc_tensor(
      affineflow::Shape{input->shape[0], channels_ * out_h * out_w}, req_grad);
  auto* op = tape->allocate_op<autograd::ops::MaxPool2dOp>(
      input, out, channels_, in_h_, in_w_, kernel_size_, stride_, pad_);
  op->forward();

  return out;
}

AvgPool2dLayer::AvgPool2dLayer(int channels, int in_h, int in_w,
                               int kernel_size, int stride, int pad)
    : channels_(channels),
      in_h_(in_h),
      in_w_(in_w),
      kernel_size_(kernel_size),
      stride_(stride),
      pad_(pad) {}

autograd::Tensor* AvgPool2dLayer::forward(autograd::Tensor* input) {
  auto* tape = autograd::Tape::get_global();
  bool req_grad = tape->record_ops_ && input->requires_grad;

  int out_h = (in_h_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int out_w = (in_w_ + 2 * pad_ - kernel_size_) / stride_ + 1;

  auto* out = tape->alloc_tensor(
      affineflow::Shape{input->shape[0], channels_ * out_h * out_w}, req_grad);
  auto* op = tape->allocate_op<autograd::ops::AvgPool2dOp>(
      input, out, channels_, in_h_, in_w_, kernel_size_, stride_, pad_);
  op->forward();

  return out;
}

}  // namespace affineflow::layers