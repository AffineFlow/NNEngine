#pragma once
#include "autograd/Op.hpp"
#include "autograd/Tensor.hpp"

namespace mlengine::autograd::ops {

class TransposeOp : public Op {
  Tensor *a_, *out_;

 public:
  TransposeOp(Tensor* a, Tensor* out) : a_(a), out_(out) {}
  void forward() override;
  void backward() override;
};

}  // namespace mlengine::autograd::ops