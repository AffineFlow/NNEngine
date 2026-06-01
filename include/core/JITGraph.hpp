#pragma once

#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "autograd/Tape.hpp"
#include "autograd/Tensor.hpp"
#include "core/DataLoader.hpp"
#include "core/Layer.hpp"
#include "core/Loss.hpp"
#include "core/Optimizer.hpp"
#include "core/Regularizer.hpp"
#include "core/Types.hpp"

namespace mlengine::core {

/**
 * @brief JIT training loop that traces one batch and replays it efficiently.
 *
 * The graph captures a representative batch once, then reuses the recorded
 * tape and parameter pointers for fast native training loops.
 */
class JITGraph {
 private:
  std::shared_ptr<Layer> model_;
  std::shared_ptr<Optimizer> optimizer_;
  std::shared_ptr<Loss> loss_fn_;
  std::shared_ptr<Regularizer> regularizer_;

  std::shared_ptr<autograd::Tape> tape_;
  autograd::Tensor* X_input_ = nullptr;
  autograd::Tensor* y_input_ = nullptr;
  autograd::Tensor* predictions_ = nullptr;
  std::vector<autograd::Tensor*> parameters_;

 public:
  /**
   * @brief Construct a compiled training graph around a model and objective.
   * @param model Differentiable network to optimize.
   * @param optimizer Update rule applied after each batch.
   * @param loss_fn Loss function used for supervision.
   * @param regularizer Optional regularization term.
   */
  JITGraph(std::shared_ptr<Layer> model, std::shared_ptr<Optimizer> optimizer,
           std::shared_ptr<Loss> loss_fn,
           std::shared_ptr<Regularizer> regularizer = nullptr)
      : model_(model),
        optimizer_(optimizer),
        loss_fn_(loss_fn),
        regularizer_(regularizer) {}

  /**
   * @brief Trace one batch, record the tape, and perform the first update.
   * @param dataloader Batch source used for tracing.
   * @return Loss value for the traced batch.
   * @note The traced graph allocates its intermediate tensors from the tape,
   *     which owns their lifetime.
   */
  float trace_batch(DataLoader& dataloader) {
    if (!dataloader.has_next()) return 0.0f;

    tape_ = std::make_shared<autograd::Tape>(true);
    autograd::TapeGuard guard(tape_.get());

    MatrixRM X_batch, y_batch;
    dataloader.next_batch(X_batch, y_batch);

    X_input_ = tape_->push_tensor(X_batch, false);
    y_input_ = tape_->push_tensor(y_batch, false);

    parameters_ = model_->parameters();
    optimizer_->set_parameters(parameters_);

    predictions_ = model_->forward(X_input_);

    float loss = loss_fn_->forward(predictions_, y_input_);
    tape_->backward();

    if (regularizer_) loss += regularizer_->apply(parameters_);
    optimizer_->step();
    return loss;
  }

  /**
   * @brief Replay the traced graph for the remaining batches in an epoch.
   * @param dataloader Batch source to consume.
   * @return Total loss and number of processed batches.
   */
  std::pair<float, size_t> fast_loop(DataLoader& dataloader) {
    float total_loss = 0.0f;
    size_t batch_count = 0;

    while (dataloader.has_next()) {
      dataloader.next_batch(X_input_->data, y_input_->data);

      optimizer_->zero_grad();
      tape_->zero_grads();

      tape_->replay_forward();
      float loss = loss_fn_->forward(predictions_, y_input_);
      tape_->replay_backward();

      if (regularizer_) loss += regularizer_->apply(parameters_);
      optimizer_->step();

      total_loss += loss;
      batch_count++;
    }
    return {total_loss, batch_count};
  }

  /**
   * @brief Evaluate the traced graph without parameter updates.
   * @param dataloader Batch source to consume.
   * @return Average loss over the available batches.
   */
  float evaluate(DataLoader& dataloader) {
    model_->train(false);

    float total_loss = 0.0f;
    size_t batch_count = 0;

    while (dataloader.has_next()) {
      dataloader.next_batch(X_input_->data, y_input_->data);
      tape_->replay_forward();
      float loss = loss_fn_->forward(predictions_, y_input_);
      if (regularizer_) loss += regularizer_->apply(parameters_);
      total_loss += loss;
      batch_count++;
    }

    model_->train(true);
    return total_loss / static_cast<float>(std::max(size_t(1), batch_count));
  }

  /**
   * @brief Train for multiple epochs with optional validation and early
   * stopping.
   * @param dataloader Training batch source.
   * @param val_dataloader Optional validation batch source.
   * @param epochs Number of epochs to run.
   * @param tol Minimum improvement required to reset the early-stopping window.
   * @param n_iter_no_change Number of epochs without improvement before stop.
   * @param verbose Whether to print progress updates.
   */
  void fast_fit(DataLoader& dataloader, DataLoader* val_dataloader, int epochs,
                float tol = 1e-4f, int n_iter_no_change = 10,
                bool verbose = true) {
    float best_loss = std::numeric_limits<float>::infinity();
    int no_improvement_count = 0;
    std::vector<MatrixRM> best_weights;

    for (int epoch = 0; epoch < epochs; ++epoch) {
      dataloader.reset();
      auto [total_loss, batches] = fast_loop(dataloader);
      float avg_train_loss =
          total_loss / static_cast<float>(std::max(size_t(1), batches));

      float metric_loss = avg_train_loss;
      if (val_dataloader) {
        val_dataloader->reset();
        metric_loss = evaluate(*val_dataloader);
      }

      if (verbose &&
          (epoch % std::max(1, epochs / 10) == 0 || epoch == epochs - 1)) {
        std::cout << "Epoch " << epoch << " | Train Loss: " << avg_train_loss;
        if (val_dataloader) std::cout << " | Val Loss: " << metric_loss;
        std::cout << std::endl;
      }

      if (best_loss - metric_loss > tol) {
        best_loss = metric_loss;
        no_improvement_count = 0;

        best_weights.clear();
        for (auto* p : parameters_) {
          best_weights.push_back(p->data);
        }
      } else if (++no_improvement_count >= n_iter_no_change) {
        if (verbose) {
          std::cout << "Early stopping triggered at epoch " << epoch
                    << ". Restoring best weights." << std::endl;
        }
        break;
      }
    }

    if (!best_weights.empty()) {
      for (size_t i = 0; i < parameters_.size(); ++i) {
        parameters_[i]->data = best_weights[i];
      }
    }
  }
};

}  // namespace mlengine::core