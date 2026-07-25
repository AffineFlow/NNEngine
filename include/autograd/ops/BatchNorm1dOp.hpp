#pragma once

#include "autograd/Op.hpp"
#include "autograd/Tensor.hpp"

namespace mlengine::autograd::ops {

/**
 * @brief Batch Normalization 1D operation primitive.
 */
class BatchNorm1dOp : public Op {
  Tensor *x_, *gamma_, *beta_, *out_;
  Tensor *running_mean_, *running_var_;
  const bool* is_training_;
  float momentum_, eps_;

  mlengine::MatrixRM x_centered_, stddev_inv_, x_hat_;

 public:
  BatchNorm1dOp(Tensor* x, Tensor* gamma, Tensor* beta, Tensor* out,
                Tensor* running_mean, Tensor* running_var, float momentum,
                float eps, const bool* is_training);

  void forward() override;
  void backward() override;
};

}  // namespace mlengine::autograd::ops