#pragma once

#include "autograd/Op.hpp"
#include "autograd/Tensor.hpp"

namespace mlengine::autograd::ops {

/**
 * @brief Dropout operation primitive for regularization.
 */
class DropoutOp : public Op {
  Tensor *a_, *out_;
  float p_;
  const bool* is_training_;
  mlengine::MatrixRM mask_;

 public:
  DropoutOp(Tensor* a, Tensor* out, float p, const bool* is_training);

  void forward() override;
  void backward() override;
};

}  // namespace mlengine::autograd::ops