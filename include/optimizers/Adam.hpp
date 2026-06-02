#pragma once

#include <iostream>
#include <vector>

#include "autograd/Tensor.hpp"
#include "core/Optimizer.hpp"

namespace mlengine::core {

/**
 * @brief Adam optimizer with first/second moment estimates.
 */
class Adam : public Optimizer {
  std::vector<autograd::MatrixRM> m_;
  std::vector<autograd::MatrixRM> v_;
  int t_ = 0;
  float beta1_ = 0.9f;
  float beta2_ = 0.999f;
  float epsilon_ = 1e-8f;

 public:
  explicit Adam(float learning_rate = 0.001f);
  void set_parameters(const std::vector<autograd::Tensor*>& params) override;
  void step() override;

  // Checkpointing implementation inside the class scope
  void save_state(std::ostream& os) const override;
  void load_state(std::istream& is) override;
};

}  // namespace mlengine::core