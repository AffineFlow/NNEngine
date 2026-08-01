#pragma once

#include <iostream>
#include <vector>

#include "autograd/Tensor.hpp"
#include "core/Optimizer.hpp"
#include "core/Types.hpp"

namespace affineflow::core {

class AdamW : public Optimizer {
  std::vector<affineflow::FlatStorage> m_;
  std::vector<affineflow::FlatStorage> v_;
  int t_ = 0;
  float beta1_ = 0.9f;
  float beta2_ = 0.999f;
  float epsilon_ = 1e-8f;
  float weight_decay_;

 public:
  explicit AdamW(float learning_rate = 0.001f, float weight_decay = 0.01f);
  void set_parameters(const std::vector<autograd::Tensor*>& params) override;
  void step() override;

  void save_state(std::ostream& os) const override;
  void load_state(std::istream& is) override;
};

}  // namespace affineflow::core