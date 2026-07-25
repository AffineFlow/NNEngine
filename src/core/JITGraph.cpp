#include "core/JITGraph.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>

#include "core/Module.hpp"

namespace mlengine::core {

JITGraph::JITGraph(std::shared_ptr<Layer> model,
                   std::shared_ptr<Optimizer> optimizer,
                   std::shared_ptr<Loss> loss_fn,
                   std::shared_ptr<Regularizer> regularizer)
    : model_(model),
      optimizer_(optimizer),
      loss_fn_(loss_fn),
      regularizer_(regularizer) {}

void JITGraph::set_scheduler(std::shared_ptr<Scheduler> scheduler) {
  scheduler_ = scheduler;
}

float JITGraph::train_step(const MatrixRM& X, const MatrixRM& y) {
  if (!X_input_) {
    X_input_ = tape_->push_tensor(X, false);
    y_input_ = tape_->push_tensor(y, false);
  } else {
    std::copy(X.data(), X.data() + X.size(), X_input_->data.data());
    std::copy(y.data(), y.data() + y.size(), y_input_->data.data());
  }
  optimizer_->zero_grad();
  tape_->zero_grads();
  tape_->replay_forward();
  float loss = loss_fn_->forward(predictions_, y_input_);
  loss_fn_->backward();
  tape_->replay_backward();
  if (regularizer_) loss += regularizer_->apply(parameters_);
  optimizer_->step();
  return loss;
}

float JITGraph::trace_batch(DataLoader& dataloader) {
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
  loss_fn_->backward();
  tape_->backward();
  if (regularizer_) loss += regularizer_->apply(parameters_);
  optimizer_->step();
  return loss;
}

std::pair<float, size_t> JITGraph::fast_loop(DataLoader& dataloader) {
  float total_loss = 0.0f;
  size_t batch_count = 0;
  MatrixRM X_batch, y_batch;
  while (dataloader.has_next()) {
    dataloader.next_batch(X_batch, y_batch);
    std::copy(X_batch.data(), X_batch.data() + X_batch.size(),
              X_input_->data.data());
    std::copy(y_batch.data(), y_batch.data() + y_batch.size(),
              y_input_->data.data());

    optimizer_->zero_grad();
    tape_->zero_grads();
    tape_->replay_forward();
    float loss = loss_fn_->forward(predictions_, y_input_);
    loss_fn_->backward();
    tape_->replay_backward();
    if (regularizer_) loss += regularizer_->apply(parameters_);
    optimizer_->step();
    total_loss += loss;
    batch_count++;
  }
  return {total_loss, batch_count};
}

float JITGraph::evaluate(DataLoader& dataloader) {
  model_->train(false);
  float total_loss = 0.0f;
  size_t batch_count = 0;
  MatrixRM X_batch, y_batch;
  while (dataloader.has_next()) {
    dataloader.next_batch(X_batch, y_batch);
    std::copy(X_batch.data(), X_batch.data() + X_batch.size(),
              X_input_->data.data());
    std::copy(y_batch.data(), y_batch.data() + y_batch.size(),
              y_input_->data.data());

    tape_->replay_forward();
    float loss = loss_fn_->forward(predictions_, y_input_);
    total_loss += loss;
    batch_count++;
  }
  model_->train(true);
  return total_loss / static_cast<float>(std::max(size_t(1), batch_count));
}

void JITGraph::fast_fit(DataLoader& dataloader, DataLoader* val_dataloader,
                        int epochs, float tol, int n_iter_no_change,
                        bool verbose) {
  float best_loss = std::numeric_limits<float>::infinity();
  int no_improvement_count = 0;
  std::vector<mlengine::FlatStorage> best_weights;

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

    if (val_dataloader) {
      if (best_loss - metric_loss > tol) {
        best_loss = metric_loss;
        no_improvement_count = 0;
        best_weights.clear();
        for (auto* p : parameters_) best_weights.push_back(p->data);
      } else if (++no_improvement_count >= n_iter_no_change) {
        if (verbose) std::cout << "Early stopping at epoch " << epoch << "\n";
        break;
      }
    }

    if (scheduler_) scheduler_->step();
  }

  if (val_dataloader && !best_weights.empty()) {
    for (size_t i = 0; i < parameters_.size(); ++i)
      parameters_[i]->data = best_weights[i];
  }
}

void JITGraph::save_checkpoint(const std::string& base_filepath) {
  if (auto* mod = dynamic_cast<core::Module*>(model_.get())) {
    mod->save_weights(base_filepath + ".weights.nne");
  }
  std::ofstream os(base_filepath + ".opt.nne", std::ios::binary);
  if (os) optimizer_->save_state(os);
}

void JITGraph::load_checkpoint(const std::string& base_filepath) {
  if (auto* mod = dynamic_cast<core::Module*>(model_.get())) {
    mod->load_weights(base_filepath + ".weights.nne");
  }
  std::ifstream is(base_filepath + ".opt.nne", std::ios::binary);
  if (is) optimizer_->load_state(is);
}
}  // namespace mlengine::core