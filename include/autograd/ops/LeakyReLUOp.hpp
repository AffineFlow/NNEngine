#pragma once
#include "autograd/Op.hpp"
#include "autograd/Tensor.hpp"

namespace affineflow::nn::autograd::ops {

/**
 * @brief Elementwise leaky rectified linear activation primitive.
 */
class LeakyReLUOp : public Op {
  Tensor *a_, *out_;
  float alpha_;

 public:
  /**
   * @brief Construct the leaky ReLU primitive.
   * @param a Input tensor.
   * @param out Output tensor written by the primitive.
   * @param alpha Negative slope for inactive elements.
   */
  LeakyReLUOp(Tensor* a, Tensor* out, float alpha)
      : a_(a), out_(out), alpha_(alpha) {}
  /**
   * @brief Apply the piecewise-linear leaky rectifier.
   */
  void forward() override;
  /**
   * @brief Scale the upstream gradient by the leaky slope on negatives.
   */
  void backward() override;
};

}  // namespace affineflow::nn::autograd::ops