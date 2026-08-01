#pragma once
#include "autograd/Op.hpp"
#include "autograd/Tensor.hpp"

namespace affineflow::autograd::ops {

class ExpOp : public Op {
  Tensor *a_, *out_;

 public:
  ExpOp(Tensor* a, Tensor* out) : a_(a), out_(out) {}
  void forward() override;
  void backward() override;
};

class LogOp : public Op {
  Tensor *a_, *out_;

 public:
  LogOp(Tensor* a, Tensor* out) : a_(a), out_(out) {}
  void forward() override;
  void backward() override;
};

}  // namespace affineflow::autograd::ops