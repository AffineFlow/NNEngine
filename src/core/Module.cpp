#include "core/Module.hpp"

#include <fstream>
#include <stdexcept>

#include "autograd/Tape.hpp"

namespace mlengine::core {

std::shared_ptr<Layer> Module::register_module(const std::string& name,
                                               std::shared_ptr<Layer> layer) {
  named_modules_.push_back({name, layer});
  return layer;
}

std::vector<autograd::Tensor*> Module::parameters() {
  std::vector<autograd::Tensor*> params;
  for (const auto& [name, m] : named_modules_) {
    auto m_params = m->parameters();
    params.insert(params.end(), m_params.begin(), m_params.end());
  }
  return params;
}

std::map<std::string, autograd::Tensor*> Module::named_parameters() {
  std::map<std::string, autograd::Tensor*> state_dict;
  for (const auto& [module_name, m] : named_modules_) {
    auto m_params = m->named_parameters();
    for (const auto& [param_name, tensor] : m_params) {
      state_dict[module_name + "." + param_name] = tensor;
    }
  }
  return state_dict;
}

void Module::train(bool mode) {
  is_training_ = mode;
  for (auto& [name, m] : named_modules_) {
    m->train(mode);
  }
}

MatrixRM Module::predict(Eigen::Ref<const MatrixRM> X) {
  bool prev_mode = is_training_;
  this->train(false);
  autograd::Tape tape(false);
  autograd::TapeGuard guard(&tape);
  autograd::Tensor* X_tensor = tape.push_tensor(X, false);
  autograd::Tensor* predictions = this->forward(X_tensor);
  this->train(prev_mode);
  return predictions->data;
}

void Module::save_weights(const std::string& filepath) {
  std::ofstream out(filepath, std::ios::binary);
  if (!out)
    throw std::runtime_error("Cannot open file for writing: " + filepath);

  auto state_dict = this->named_parameters();
  size_t num_params = state_dict.size();
  out.write(reinterpret_cast<const char*>(&num_params), sizeof(size_t));

  for (const auto& [name, tensor] : state_dict) {
    size_t name_len = name.size();
    out.write(reinterpret_cast<const char*>(&name_len), sizeof(size_t));
    out.write(name.c_str(), name_len);

    size_t rows = tensor->data.rows();
    size_t cols = tensor->data.cols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(tensor->data.data()),
              rows * cols * sizeof(float));
  }
}

void Module::load_weights(const std::string& filepath) {
  std::ifstream in(filepath, std::ios::binary);
  if (!in)
    throw std::runtime_error("Cannot open file for reading: " + filepath);

  auto state_dict = this->named_parameters();
  size_t num_params;
  in.read(reinterpret_cast<char*>(&num_params), sizeof(size_t));

  for (size_t i = 0; i < num_params; ++i) {
    size_t name_len;
    in.read(reinterpret_cast<char*>(&name_len), sizeof(size_t));
    std::string name(name_len, '\0');
    in.read(&name[0], name_len);

    size_t rows, cols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&cols), sizeof(size_t));

    auto it = state_dict.find(name);
    if (it == state_dict.end()) {
      in.seekg(rows * cols * sizeof(float), std::ios::cur);
      continue;
    }

    auto* p = it->second;
    if (rows != p->data.rows() || cols != p->data.cols()) {
      throw std::runtime_error(
          "Tensor shape mismatch in saved weights for layer: " + name);
    }
    in.read(reinterpret_cast<char*>(p->data.data()),
            rows * cols * sizeof(float));
  }
}

}  // namespace mlengine::core