#pragma once

#include <map>
#include <string>
#include <vector>

#include "autograd/Tensor.hpp"
#include "core/Layer.hpp"

namespace affineflow::nn::layers {

/**
 * @brief Applies Batch Normalization over a 1D input tensor.
 */
class BatchNorm1dLayer : public core::Layer {
  autograd::Tensor gamma_;
  autograd::Tensor beta_;
  autograd::Tensor running_mean_;
  autograd::Tensor running_var_;
  float momentum_, eps_;

 public:
  /**
   * @brief Construct a Batch Normalization layer.
   * @param num_features Number of features in the input.
   * @param eps Value added to denominator for numerical stability.
   * @param momentum Value used for running mean and variance computation.
   */
  BatchNorm1dLayer(int num_features, float eps = 1e-5f, float momentum = 0.1f);

  autograd::Tensor* forward(autograd::Tensor* input) override;
  std::vector<autograd::Tensor*> parameters() override;
  std::map<std::string, autograd::Tensor*> named_parameters() override;
};

}  // namespace affineflow::nn::layers