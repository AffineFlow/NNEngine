#include "core/DataLoader.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "core/Random.hpp"

namespace mlengine::core {

DataLoader::DataLoader(const autograd::Tensor& X, const autograd::Tensor& y,
                       size_t batch_size, bool shuffle, bool drop_last)
    : X_(X),
      y_(y),
      batch_size_(batch_size),
      shuffle_(shuffle),
      drop_last_(drop_last),
      current_idx_(0) {
  if (X_.shape.empty() || y_.shape.empty() || X_.shape[0] != y_.shape[0]) {
    throw std::runtime_error("Features and Targets must have same row count.");
  }
  indices_.resize(X_.shape[0]);
  std::iota(indices_.begin(), indices_.end(), 0);
  reset();
}

void DataLoader::reset() {
  current_idx_ = 0;
  if (shuffle_ && !indices_.empty()) {
    std::shuffle(indices_.begin(), indices_.end(), core::rng());
  }
}

bool DataLoader::has_next() const {
  if (drop_last_) return current_idx_ + batch_size_ <= indices_.size();
  return current_idx_ < indices_.size();
}

void DataLoader::next_batch(autograd::Tensor& X_batch,
                            autograd::Tensor& y_batch) {
  if (!has_next()) return;
  size_t remaining = indices_.size() - current_idx_;
  size_t actual_batch_size = std::min(batch_size_, remaining);

  mlengine::Shape expected_x = X_.shape;
  expected_x[0] = actual_batch_size;
  if (X_batch.shape != expected_x) {
    X_batch.resize(expected_x);
  }

  mlengine::Shape expected_y = y_.shape;
  expected_y[0] = actual_batch_size;
  if (y_batch.shape != expected_y) {
    y_batch.resize(expected_y);
  }

  Eigen::Index x_row_size = mlengine::compute_size(X_.shape) / X_.shape[0];
  Eigen::Index y_row_size = mlengine::compute_size(y_.shape) / y_.shape[0];

  for (size_t i = 0; i < actual_batch_size; ++i) {
    std::copy_n(X_.data.data() + (indices_[current_idx_ + i] * x_row_size),
                x_row_size, X_batch.data.data() + (i * x_row_size));
    std::copy_n(y_.data.data() + (indices_[current_idx_ + i] * y_row_size),
                y_row_size, y_batch.data.data() + (i * y_row_size));
  }
  current_idx_ += actual_batch_size;
}

}  // namespace mlengine::core