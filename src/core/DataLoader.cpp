#include "core/DataLoader.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "core/Random.hpp"

namespace mlengine::core {

DataLoader::DataLoader(const MatrixRM& X, const MatrixRM& y, size_t batch_size,
                       bool shuffle, bool drop_last)
    : X_(X),
      y_(y),
      batch_size_(batch_size),
      shuffle_(shuffle),
      drop_last_(drop_last),
      current_idx_(0) {
  if (X_.rows() != y_.rows()) {
    throw std::runtime_error("Features and Targets must have same row count.");
  }
  indices_.resize(X_.rows());
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

void DataLoader::next_batch(MatrixRM& X_batch, MatrixRM& y_batch) {
  if (!has_next()) return;
  size_t remaining = indices_.size() - current_idx_;
  size_t actual_batch_size = std::min(batch_size_, remaining);

  if (X_batch.rows() != actual_batch_size || X_batch.cols() != X_.cols()) {
    X_batch.resize(actual_batch_size, X_.cols());
  }
  if (y_batch.rows() != actual_batch_size || y_batch.cols() != y_.cols()) {
    y_batch.resize(actual_batch_size, y_.cols());
  }

  // Zero-allocation indexed gather
  for (size_t i = 0; i < actual_batch_size; ++i) {
    X_batch.row(i) = X_.row(indices_[current_idx_ + i]);
    y_batch.row(i) = y_.row(indices_[current_idx_ + i]);
  }
  current_idx_ += actual_batch_size;
}

}  // namespace mlengine::core