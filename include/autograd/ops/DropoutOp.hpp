#pragma once

#include "autograd/Op.hpp"
#include "autograd/Tensor.hpp"

namespace affineengine::autograd::ops {

class DropoutOp : public Op {
  Tensor *a_, *out_;
  float p_;
  const bool* is_training_;
  affineengine::FlatStorage mask_;

 public:
  DropoutOp(Tensor* a, Tensor* out, float p, const bool* is_training);

  void forward() override;
  void backward() override;
};

}  // namespace affineengine::autograd::ops