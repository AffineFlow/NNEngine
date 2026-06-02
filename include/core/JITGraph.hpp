#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "autograd/Tape.hpp"
#include "autograd/Tensor.hpp"
#include "core/DataLoader.hpp"
#include "core/Layer.hpp"
#include "core/Loss.hpp"
#include "core/Optimizer.hpp"
#include "core/Regularizer.hpp"
#include "core/Scheduler.hpp"

namespace mlengine::core {

/**
 * @brief Just-In-Time compiled training loop executor.
 * * Traces a computational graph during the first batch and executes
 * a highly optimized, zero-allocation native C++ replay loop for all subsequent
 * batches.
 */
class JITGraph {
 private:
  std::shared_ptr<Layer> model_;
  std::shared_ptr<Optimizer> optimizer_;
  std::shared_ptr<Loss> loss_fn_;
  std::shared_ptr<Regularizer> regularizer_;
  std::shared_ptr<Scheduler> scheduler_ = nullptr;

  std::shared_ptr<autograd::Tape> tape_;
  autograd::Tensor* X_input_ = nullptr;
  autograd::Tensor* y_input_ = nullptr;
  autograd::Tensor* predictions_ = nullptr;
  std::vector<autograd::Tensor*> parameters_;

 public:
  JITGraph(std::shared_ptr<Layer> model, std::shared_ptr<Optimizer> optimizer,
           std::shared_ptr<Loss> loss_fn,
           std::shared_ptr<Regularizer> regularizer = nullptr);

  /** @brief Attach a learning rate scheduler to the training loop. */
  void set_scheduler(std::shared_ptr<Scheduler> scheduler);

  /** @brief Execute a single optimization step without data loaders. */
  float train_step(const MatrixRM& X, const MatrixRM& y);

  /** @brief Trace the computational graph for the first batch. */
  float trace_batch(DataLoader& dataloader);

  /** @brief Replay the traced graph natively across remaining batches. */
  std::pair<float, size_t> fast_loop(DataLoader& dataloader);

  /** @brief Run an evaluation pass without updating parameters. */
  float evaluate(DataLoader& dataloader);

  /** @brief Execute the full multi-epoch JIT training loop. */
  void fast_fit(DataLoader& dataloader, DataLoader* val_dataloader, int epochs,
                float tol = 1e-4f, int n_iter_no_change = 10,
                bool verbose = true);

  /** @brief Native serialization for weights and optimizer moments. */
  void save_checkpoint(const std::string& base_filepath);

  /** @brief Native deserialization for weights and optimizer moments. */
  void load_checkpoint(const std::string& base_filepath);
};

}  // namespace mlengine::core