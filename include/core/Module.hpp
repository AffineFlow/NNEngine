#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "autograd/Tensor.hpp"
#include "core/Layer.hpp"

namespace mlengine::core {

/**
 * @brief Composite layer that manages submodules and persistence utilities.
 */
class Module : public Layer {
 protected:
  std::vector<std::pair<std::string, std::shared_ptr<Layer>>> named_modules_;

 public:
  virtual ~Module() = default;

  /**
   * @brief Register a child layer with a specific string identifier.
   * @param name The identifier for the layer (e.g., "fc1").
   * @param layer Layer to append to the module hierarchy.
   * @return The same layer pointer for fluent composition.
   */
  std::shared_ptr<Layer> register_module(const std::string& name,
                                         std::shared_ptr<Layer> layer);

  /**
   * @brief Collect all trainable tensors from child modules.
   * @return Flat list of parameter tensors.
   */
  std::vector<autograd::Tensor*> parameters() override;

  /**
   * @brief Collect recursively prefixed named parameters.
   * @return Key-value map of parameter strings to tensors.
   */
  std::map<std::string, autograd::Tensor*> named_parameters() override;

  /**
   * @brief Recursively set the training mode for this module and all children.
   * @param mode True for training, false for evaluation.
   */
  void train(bool mode = true) override;

  /**
   * @brief Execute a forward pass without tracking gradients.
   * @param X Input matrix.
   * @return The predicted output matrix.
   */
  MatrixRM predict(Eigen::Ref<const MatrixRM> X);

  /**
   * @brief Serialize model parameters to a binary .nne file.
   * @param filepath Target file destination.
   */
  void save_weights(const std::string& filepath);

  /**
   * @brief Load and map parameters from a binary .nne file robustly.
   * @param filepath Source file.
   */
  void load_weights(const std::string& filepath);
};

}  // namespace mlengine::core