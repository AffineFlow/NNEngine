#pragma once
#include "autograd/Op.hpp"
#include "autograd/Tensor.hpp"

namespace affineflow::nn::autograd::ops {

class AddOp : public Op {
  Tensor *a_, *b_, *out_;

 public:
  AddOp(Tensor* a, Tensor* b, Tensor* out) : a_(a), b_(b), out_(out) {}
  void forward() override;
  void backward() override;
};

}  // namespace affineflow::nn::autograd::ops