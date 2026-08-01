#pragma once
#include "autograd/Op.hpp"
#include "autograd/Tensor.hpp"

namespace affineflow::autograd::ops {

class MulOp : public Op {
  Tensor *a_, *b_, *out_;

 public:
  MulOp(Tensor* a, Tensor* b, Tensor* out) : a_(a), b_(b), out_(out) {}
  void forward() override;
  void backward() override;
};

}  // namespace affineflow::autograd::ops