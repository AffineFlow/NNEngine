#pragma once
#include <vector>

#include "autograd/Tensor.hpp"
#include "core/Types.hpp"

namespace mlengine::core {

class DataLoader {
 private:
  autograd::Tensor X_;
  autograd::Tensor y_;
  size_t batch_size_;
  bool shuffle_;
  bool drop_last_;
  std::vector<int> indices_;
  size_t current_idx_;

 public:
  DataLoader(const autograd::Tensor& X, const autograd::Tensor& y,
             size_t batch_size, bool shuffle = true, bool drop_last = false);
  void reset();
  bool has_next() const;
  void next_batch(autograd::Tensor& X_batch, autograd::Tensor& y_batch);
};

}  // namespace mlengine::core