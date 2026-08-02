#include "core/Module.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>

#include "autograd/Tape.hpp"

namespace affineflow::core {

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

autograd::Tensor Module::predict(const autograd::Tensor& X) {
  bool prev_mode = is_training_;
  this->train(false);
  autograd::Tape tape(false);
  autograd::TapeGuard guard(&tape);
  autograd::Tensor* X_tensor = tape.push_tensor(X, false);
  autograd::Tensor* predictions = this->forward(X_tensor);
  this->train(prev_mode);

  autograd::Tensor out(predictions->shape, false);
  std::copy_n(predictions->data.data(), predictions->data.size(),
              out.data.data());
  return out;
}

void Module::save_weights(const std::string& filepath) {
  std::ofstream out(filepath, std::ios::binary);
  if (!out)
    throw std::runtime_error("Cannot open file for writing: " + filepath);

  auto state_dict = this->named_parameters();
  uint32_t num_params = static_cast<uint32_t>(state_dict.size());
  out.write(reinterpret_cast<const char*>(&num_params), sizeof(uint32_t));

  for (const auto& [name, tensor] : state_dict) {
    uint32_t name_len = static_cast<uint32_t>(name.size());
    out.write(reinterpret_cast<const char*>(&name_len), sizeof(uint32_t));
    out.write(name.c_str(), name_len);

    uint32_t shape_size = static_cast<uint32_t>(tensor->shape.size());
    out.write(reinterpret_cast<const char*>(&shape_size), sizeof(uint32_t));
    for (auto dim : tensor->shape) {
      int32_t d = static_cast<int32_t>(dim);
      out.write(reinterpret_cast<const char*>(&d), sizeof(int32_t));
    }
    out.write(reinterpret_cast<const char*>(tensor->data.data()),
              tensor->data.size() * sizeof(float));
  }
}

void Module::load_weights(const std::string& filepath) {
  std::ifstream in(filepath, std::ios::binary);
  if (!in)
    throw std::runtime_error("Cannot open file for reading: " + filepath);

  auto state_dict = this->named_parameters();
  uint32_t num_params;
  in.read(reinterpret_cast<char*>(&num_params), sizeof(uint32_t));

  for (uint32_t i = 0; i < num_params; ++i) {
    uint32_t name_len;
    in.read(reinterpret_cast<char*>(&name_len), sizeof(uint32_t));
    std::string name(name_len, '\0');
    in.read(&name[0], name_len);

    uint32_t shape_size;
    in.read(reinterpret_cast<char*>(&shape_size), sizeof(uint32_t));
    std::vector<Eigen::Index> loaded_shape(shape_size);
    Eigen::Index total_size = 1;
    for (uint32_t j = 0; j < shape_size; ++j) {
      int32_t dim;
      in.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
      loaded_shape[j] = static_cast<Eigen::Index>(dim);
      total_size *= loaded_shape[j];
    }

    auto it = state_dict.find(name);
    if (it == state_dict.end()) {
      in.seekg(total_size * sizeof(float), std::ios::cur);
      continue;
    }

    auto* p = it->second;
    if (p->shape != loaded_shape) {
      throw std::runtime_error(
          "Tensor shape mismatch in saved weights for layer: " + name);
    }
    in.read(reinterpret_cast<char*>(p->data.data()),
            total_size * sizeof(float));
  }
}

}  // namespace affineflow::core