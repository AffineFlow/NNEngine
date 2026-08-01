#pragma once
#include "autograd/Op.hpp"
#include "autograd/Tensor.hpp"

namespace affineengine::autograd::ops {

/**
 * @brief Matrix multiplication primitive for dense layers.
 */
class MatMulOp : public Op {
  Tensor *a_, *b_, *out_;

 public:
  /**
   * @brief Construct the matrix multiplication primitive.
   * @param a Left-hand matrix operand.
   * @param b Right-hand matrix operand.
   * @param out Output tensor written by the primitive.
   */
  MatMulOp(Tensor* a, Tensor* b, Tensor* out) : a_(a), b_(b), out_(out) {}
  /**
   * @brief Compute the dense matrix product.
   */
  void forward() override;
  /**
   * @brief Accumulate gradients using the transpose identities.
   */
  void backward() override;
};

}  // namespace affineengine::autograd::ops