#pragma once

#include <Eigen/Core>
#include <map>
#include <string>
#include <vector>

#include "autograd/Tape.hpp"
#include "core/Types.hpp"

namespace mlengine::core {

using MatrixRM = mlengine::MatrixRM;

/**
 * @brief Abstract building block for differentiable model components.
 */
class Layer {
 public:
  virtual ~Layer() = default;

  /**
   * @brief Run the layer on tape-owned input storage.
   * @param input Input tensor to transform.
   * @return Pointer to the tape-owned output tensor.
   * @note Implementations must allocate outputs from the active global tape.
   */
  virtual autograd::Tensor* forward(autograd::Tensor* input) = 0;

  /**
   * @brief Return mutable pointers to trainable parameters.
   * @return Trainable tensors owned by the layer.
   */
  virtual std::vector<autograd::Tensor*> parameters() { return {}; }

  /**
   * @brief Return named trainable parameters for serialization mapping.
   * @return Map of string names to trainable tensors.
   */
  virtual std::map<std::string, autograd::Tensor*> named_parameters() {
    return {};
  }
};

}  // namespace mlengine::core