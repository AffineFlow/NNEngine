#include "core/JITGraph.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>

#include "core/Module.hpp"

namespace affineflow::nn::core {

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

float JITGraph::train_step(const autograd::Tensor& X,
                           const autograd::Tensor& y) {
  if (!X_input_) {
    X_input_ = tape_->push_tensor(X, false);
    y_input_ = tape_->push_tensor(y, false);
  } else {
    std::copy(X.data.data(), X.data.data() + X.data.size(),
              X_input_->data.data());
    std::copy(y.data.data(), y.data.data() + y.data.size(),
              y_input_->data.data());
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
  if (!dataloader.has_next()) {
    throw std::runtime_error(
        "DataLoader exhausted before JIT trace. Reset it first.");
  }
  tape_ = std::make_shared<autograd::Tape>(true);
  autograd::TapeGuard guard(tape_.get());

  autograd::Tensor X_batch(affineflow::nn::Shape{0});
  autograd::Tensor y_batch(affineflow::nn::Shape{0});
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

  optimizer_->zero_grad();

  return loss;
}

std::pair<float, size_t> JITGraph::fast_loop(DataLoader& dataloader) {
  float total_loss = 0.0f;
  size_t batch_count = 0;

  while (dataloader.has_next()) {
    dataloader.next_batch(*X_input_, *y_input_);

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

  while (dataloader.has_next()) {
    dataloader.next_batch(*X_input_, *y_input_);

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
  std::vector<affineflow::nn::FlatStorage> best_weights;

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
  std::ofstream out(base_filepath + ".weights.nne", std::ios::binary);
  if (out) {
    auto state_dict = model_->named_parameters();
    uint32_t num_params = static_cast<uint32_t>(state_dict.size());
    out.write(reinterpret_cast<const char*>(&num_params), sizeof(uint32_t));

    for (const auto& [name, tensor] : state_dict) {
      uint32_t name_len = static_cast<uint32_t>(name.size());
      out.write(reinterpret_cast<const char*>(&name_len), sizeof(uint32_t));
      out.write(name.c_str(), name_len);

      uint32_t shape_size = static_cast<uint32_t>(tensor->shape.size());
      out.write(reinterpret_cast<const char*>(&shape_size), sizeof(uint32_t));
      for (auto dim : tensor->shape) {
        int32_t d = static_cast<int32_t>(dim);
        out.write(reinterpret_cast<const char*>(&d), sizeof(int32_t));
      }
      out.write(reinterpret_cast<const char*>(tensor->data.data()),
                tensor->data.size() * sizeof(float));
    }
  }

  std::ofstream os(base_filepath + ".opt.nne", std::ios::binary);
  if (os) optimizer_->save_state(os);
}

void JITGraph::load_checkpoint(const std::string& base_filepath) {
  std::ifstream in(base_filepath + ".weights.nne", std::ios::binary);
  if (in) {
    auto state_dict = model_->named_parameters();
    uint32_t num_params;
    in.read(reinterpret_cast<char*>(&num_params), sizeof(uint32_t));

    for (uint32_t i = 0; i < num_params; ++i) {
      uint32_t name_len;
      in.read(reinterpret_cast<char*>(&name_len), sizeof(uint32_t));
      std::string name(name_len, '\0');
      in.read(&name[0], name_len);

      uint32_t shape_size;
      in.read(reinterpret_cast<char*>(&shape_size), sizeof(uint32_t));
      std::vector<Eigen::Index> loaded_shape(shape_size);
      Eigen::Index total_size = 1;
      for (uint32_t j = 0; j < shape_size; ++j) {
        int32_t dim;
        in.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
        loaded_shape[j] = static_cast<Eigen::Index>(dim);
        total_size *= loaded_shape[j];
      }

      auto it = state_dict.find(name);
      if (it != state_dict.end() && it->second->shape == loaded_shape) {
        in.read(reinterpret_cast<char*>(it->second->data.data()),
                total_size * sizeof(float));
      } else {
        in.seekg(total_size * sizeof(float), std::ios::cur);
      }
    }
  }

  std::ifstream is(base_filepath + ".opt.nne", std::ios::binary);
  if (is) optimizer_->load_state(is);
}
}  // namespace affineflow::nn::core