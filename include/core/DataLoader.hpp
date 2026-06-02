#pragma once

#include <vector>

#include "core/Types.hpp"

namespace mlengine::core {

/**
 * @brief Mini-batch iterator over dense feature and target matrices.
 *
 * Handles zero-copy slicing of the dataset and optional epoch reshuffling.
 */
class DataLoader {
 private:
  MatrixRM X_;
  MatrixRM y_;
  MatrixRM X_shuffled_;
  MatrixRM y_shuffled_;
  size_t batch_size_;
  bool shuffle_;
  bool drop_last_;
  std::vector<int> indices_;
  size_t current_idx_;

 public:
  /**
   * @brief Build a loader from in-memory feature and target matrices.
   * @param X Feature matrix with samples in rows.
   * @param y Target matrix with matching row count.
   * @param batch_size Number of samples per batch.
   * @param shuffle Whether to shuffle rows at the start of each epoch.
   * @param drop_last Whether to drop the final partial batch.
   */
  DataLoader(const MatrixRM& X, const MatrixRM& y, size_t batch_size,
             bool shuffle = true, bool drop_last = false);

  /**
   * @brief Reset iteration state and reshuffle if enabled.
   */
  void reset();

  /**
   * @brief Check whether another batch is available.
   * @return True if a subsequent call to next_batch can fill output buffers.
   */
  bool has_next() const;

  /**
   * @brief Copy the next batch into caller-provided buffers.
   * @param X_batch Output feature buffer.
   * @param y_batch Output target buffer.
   */
  void next_batch(MatrixRM& X_batch, MatrixRM& y_batch);
};

}  // namespace mlengine::core