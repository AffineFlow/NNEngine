#pragma once
#include <vector>

#include "core/Types.hpp"

namespace mlengine::core {

class DataLoader {
 private:
  MatrixRM X_;
  MatrixRM y_;
  size_t batch_size_;
  bool shuffle_;
  bool drop_last_;
  std::vector<int> indices_;
  size_t current_idx_;

 public:
  DataLoader(const MatrixRM& X, const MatrixRM& y, size_t batch_size,
             bool shuffle = true, bool drop_last = false);
  void reset();
  bool has_next() const;
  void next_batch(MatrixRM& X_batch, MatrixRM& y_batch);
};

}  // namespace mlengine::core