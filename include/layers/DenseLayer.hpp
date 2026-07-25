#pragma once
#include <vector>

#include "autograd/Tensor.hpp"
#include "core/Layer.hpp"

namespace mlengine::layers {

/**
 * @brief Fully connected affine layer with learned weights and bias.
 */
class DenseLayer : public core::Layer {
 public:
  /**
   * @brief Create a dense layer with Glorot-style uniform initialization.
   * @param input_dim Number of input features.
   * @param output_dim Number of output features.
   */
  DenseLayer(int input_dim, int output_dim);

  /**
   * @brief Apply the affine transform and record the corresponding tape ops.
   * @param input Input activation tensor.
   * @return Pointer to the tape-owned output tensor.
   * @note The returned tensor is arena-allocated and owned by the tape, not
   *     the caller.
   */
  autograd::Tensor* forward(autograd::Tensor* input) override;

  /**
   * @brief Return the learned weight and bias tensors.
   * @return Mutable parameter pointers for optimization.
   */
  std::vector<autograd::Tensor*> parameters() override;

  /**
   * @brief Return the learned weight and bias tensors with their local names.
   * @return Map of local parameter names to tensors.
   */
  std::map<std::string, autograd::Tensor*> named_parameters() override {
    return {{"weight", &weights_}, {"bias", &bias_}};
  }

 private:
  autograd::Tensor weights_;
  autograd::Tensor bias_;
};

}  // namespace mlengine::layers