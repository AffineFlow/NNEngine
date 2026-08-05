#pragma once

#include <iostream>
#include <vector>

#include "autograd/Tensor.hpp"

namespace affineflow::nn::core {

/**
 * @brief Base class for parameter update rules.
 */
class Optimizer {
 protected:
  std::vector<autograd::Tensor*> parameters_;
  float lr_;

 public:
  explicit Optimizer(float learning_rate) : lr_(learning_rate) {}
  virtual ~Optimizer() = default;

  virtual void set_parameters(const std::vector<autograd::Tensor*>& params) {
    parameters_ = params;
  }

  virtual void step() = 0;

  void zero_grad() {
    for (auto* p : parameters_) {
      p->zero_grad();
    }
  }

  // State checkpointing methods
  virtual void save_state(std::ostream& os) const {}
  virtual void load_state(std::istream& is) {}

  virtual float get_lr() const { return lr_; }
  virtual void set_lr(float lr) { lr_ = lr; }
};

}  // namespace affineflow::nn::core