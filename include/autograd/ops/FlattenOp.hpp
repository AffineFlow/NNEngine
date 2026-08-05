#pragma once
#include "autograd/Op.hpp"
#include "autograd/Tensor.hpp"

namespace affineflow::nn::autograd::ops {

class FlattenOp : public Op {
  Tensor *x_, *out_;

 public:
  FlattenOp(Tensor* x, Tensor* out) : x_(x), out_(out) {}
  void forward() override;
  void backward() override;
};

}  // namespace affineflow::nn::autograd::ops