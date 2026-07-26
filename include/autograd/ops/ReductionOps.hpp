#pragma once
#include "autograd/Op.hpp"
#include "autograd/Tensor.hpp"

namespace mlengine::autograd::ops {

class SumOp : public Op {
  Tensor *a_, *out_;

 public:
  SumOp(Tensor* a, Tensor* out) : a_(a), out_(out) {}
  void forward() override;
  void backward() override;
};

class MeanOp : public Op {
  Tensor *a_, *out_;

 public:
  MeanOp(Tensor* a, Tensor* out) : a_(a), out_(out) {}
  void forward() override;
  void backward() override;
};

}  // namespace mlengine::autograd::ops