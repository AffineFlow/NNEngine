#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "autograd/Tensor.hpp"
#include "core/Layer.hpp"

namespace mlengine::core {

class Module : public Layer {
 protected:
  std::vector<std::pair<std::string, std::shared_ptr<Layer>>> named_modules_;

 public:
  virtual ~Module() = default;

  std::shared_ptr<Layer> register_module(const std::string& name,
                                         std::shared_ptr<Layer> layer);

  std::vector<autograd::Tensor*> parameters() override;
  std::map<std::string, autograd::Tensor*> named_parameters() override;
  void train(bool mode = true) override;

  autograd::Tensor predict(const autograd::Tensor& X);

  void save_weights(const std::string& filepath);
  void load_weights(const std::string& filepath);
};

}  // namespace mlengine::core