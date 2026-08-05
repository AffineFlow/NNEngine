#pragma once

#include <vector>

#include "autograd/Tensor.hpp"
#include "core/Regularizer.hpp"

namespace affineflow::nn::core {

/**
 * @brief L2 weight decay regularizer.
 */
class L2Regularizer : public Regularizer {
  float l2_;

 public:
  /**
   * @brief Construct the regularizer with the desired penalty strength.
   * @param l2 Coefficient applied to the squared-norm penalty.
   */
  explicit L2Regularizer(float l2 = 0.0001f);

  /**
   * @brief Add the L2 penalty and its gradient contribution in place.
   * @param parameters Mutable parameter pointers to regularize.
   * @return Half-scaled L2 penalty over all trainable parameters.
   */
  float apply(const std::vector<autograd::Tensor*>& parameters) override;
};

}  // namespace affineflow::nn::core
