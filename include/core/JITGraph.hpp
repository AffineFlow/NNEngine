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

namespace affineengine::core {

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

  void set_scheduler(std::shared_ptr<Scheduler> scheduler);
  float train_step(const autograd::Tensor& X, const autograd::Tensor& y);
  float trace_batch(DataLoader& dataloader);
  std::pair<float, size_t> fast_loop(DataLoader& dataloader);
  float evaluate(DataLoader& dataloader);
  void fast_fit(DataLoader& dataloader, DataLoader* val_dataloader, int epochs,
                float tol = 1e-4f, int n_iter_no_change = 10,
                bool verbose = true);
  void save_checkpoint(const std::string& base_filepath);
  void load_checkpoint(const std::string& base_filepath);
};

}  // namespace affineengine::core