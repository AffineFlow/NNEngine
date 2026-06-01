#pragma once

#include <fstream>
#include <memory>
#include <stdexcept>
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
  std::vector<std::shared_ptr<Layer>> modules_;

 public:
  virtual ~Module() = default;

  /**
   * @brief Register a child layer and retain shared ownership.
   * @param layer Layer to append to the module hierarchy.
   * @return The same layer pointer for fluent composition.
   */
  std::shared_ptr<Layer> add_module(std::shared_ptr<Layer> layer) {
    modules_.push_back(layer);
    return layer;
  }

  /**
   * @brief Collect all trainable tensors from child modules.
   * @return Mutable pointers to the parameters owned by sublayers.
   */
  std::vector<autograd::Tensor*> parameters() override {
    std::vector<autograd::Tensor*> params;
    params.reserve(modules_.size() * 2);

    for (const auto& m : modules_) {
      auto m_params = m->parameters();
      params.insert(params.end(), m_params.begin(), m_params.end());
    }
    return params;
  }

  /**
   * @brief Run inference without recording autograd operations.
   * @param X Input matrix in row-major format.
   * @return Predicted output matrix.
   * @note The temporary input tensor is allocated from a local tape and is
   *     owned by that tape, not the caller.
   */
  MatrixRM predict(Eigen::Ref<const MatrixRM> X) {
    autograd::Tape tape(false);
    autograd::Tape::set_global(&tape);
    autograd::Tensor* X_tensor = tape.push_tensor(X, false);
    autograd::Tensor* predictions = this->forward(X_tensor);
    autograd::Tape::set_global(nullptr);
    return predictions->data;
  }

  /**
   * @brief Serialize all model parameters to a compact binary blob.
   * @param filepath Destination path for the serialized weights.
   */
  void save_weights(const std::string& filepath) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out)
      throw std::runtime_error("Cannot open file for writing: " + filepath);

    auto params = this->parameters();
    size_t num_params = params.size();
    out.write(reinterpret_cast<const char*>(&num_params), sizeof(size_t));

    for (auto* p : params) {
      size_t rows = p->data.rows();
      size_t cols = p->data.cols();
      out.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
      out.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));

      out.write(reinterpret_cast<const char*>(p->data.data()),
                rows * cols * sizeof(float));
    }
  }

  /**
   * @brief Restore model parameters from a previously saved binary blob.
   * @param filepath Source path for the serialized weights.
   */
  void load_weights(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in)
      throw std::runtime_error("Cannot open file for reading: " + filepath);

    auto params = this->parameters();
    size_t num_params;
    in.read(reinterpret_cast<char*>(&num_params), sizeof(size_t));

    if (num_params != params.size()) {
      throw std::runtime_error("Parameter count mismatch in saved weights.");
    }

    for (auto* p : params) {
      size_t rows, cols;
      in.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
      in.read(reinterpret_cast<char*>(&cols), sizeof(size_t));

      if (rows != p->data.rows() || cols != p->data.cols()) {
        throw std::runtime_error("Tensor shape mismatch in saved weights.");
      }

      in.read(reinterpret_cast<char*>(p->data.data()),
              rows * cols * sizeof(float));
    }
  }
};

}  // namespace mlengine::core