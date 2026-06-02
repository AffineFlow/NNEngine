#pragma once

#include <memory>

#include "core/Optimizer.hpp"

namespace mlengine::core {

class Scheduler {
 protected:
  std::shared_ptr<Optimizer> opt_;

 public:
  explicit Scheduler(std::shared_ptr<Optimizer> opt) : opt_(opt) {}
  virtual ~Scheduler() = default;
  virtual void step() = 0;
};

class StepLR : public Scheduler {
  int step_size_;
  float gamma_;
  int epoch_ = 0;

 public:
  StepLR(std::shared_ptr<Optimizer> opt, int step_size, float gamma = 0.1f)
      : Scheduler(opt), step_size_(step_size), gamma_(gamma) {}

  void step() override {
    epoch_++;
    if (epoch_ % step_size_ == 0) {
      opt_->set_lr(opt_->get_lr() * gamma_);
    }
  }
};

}  // namespace mlengine::core