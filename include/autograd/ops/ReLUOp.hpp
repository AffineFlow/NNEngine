#pragma once
#include "autograd/Op.hpp"
#include "autograd/Tensor.hpp"

namespace affineflow::autograd::ops {

/**
 * @brief Elementwise rectified linear activation primitive.
 */
class ReLUOp : public Op {
  Tensor *a_, *out_;

 public:
  /**
   * @brief Construct the ReLU primitive.
   * @param a Input tensor.
   * @param out Output tensor written by the primitive.
   */
  ReLUOp(Tensor* a, Tensor* out) : a_(a), out_(out) {}
  /**
   * @brief Apply the piecewise-linear rectifier.
   */
  void forward() override;
  /**
   * @brief Mask the upstream gradient where the activation was non-positive.
   */
  void backward() override;
};

}  // namespace affineflow::autograd::ops