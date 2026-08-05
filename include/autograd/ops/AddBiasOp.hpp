#pragma once
#include "autograd/Op.hpp"
#include "autograd/Tensor.hpp"

namespace affineflow::nn::autograd::ops {

/**
 * @brief Broadcast-add a row bias to a batch of activations.
 */
class AddBiasOp : public Op {
  Tensor *a_, *b_, *out_;

 public:
  /**
   * @brief Construct the bias-add primitive.
   * @param a Input activation tensor.
   * @param b Bias tensor broadcast across rows.
   * @param out Output tensor written by the primitive.
   */
  AddBiasOp(Tensor* a, Tensor* b, Tensor* out) : a_(a), b_(b), out_(out) {}
  /**
   * @brief Compute rowwise addition with bias broadcasting.
   */
  void forward() override;
  /**
   * @brief Propagate gradients to the input activation and bias tensors.
   */
  void backward() override;
};

}  // namespace affineflow::nn::autograd::ops