#pragma once

#include "autograd/Op.hpp"
#include "autograd/Tensor.hpp"

namespace affineflow::nn::autograd::ops {

class DropoutOp : public Op {
  Tensor *a_, *out_;
  float p_;
  const bool* is_training_;
  affineflow::nn::FlatStorage mask_;

 public:
  DropoutOp(Tensor* a, Tensor* out, float p, const bool* is_training);

  void forward() override;
  void backward() override;
};

}  // namespace affineflow::nn::autograd::ops