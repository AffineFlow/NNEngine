#include "layers/DenseLayer.hpp"

#include <cmath>
#include <random>

#include "autograd/Tape.hpp"
#include "autograd/ops/AddBiasOp.hpp"
#include "autograd/ops/MatMulOp.hpp"
#include "core/Random.hpp"

namespace mlengine::layers {

DenseLayer::DenseLayer(int input_dim, int output_dim)
    : weights_(mlengine::Shape{input_dim, output_dim}, true, true),
      bias_(mlengine::Shape{output_dim}, true, false) {
  // PyTorch default Linear initialization
  float limit = 1.0f / std::sqrt(static_cast<float>(input_dim));
  std::uniform_real_distribution<float> dist(-limit, limit);
  auto& gen = core::rng();

  float* w_ptr = weights_.data.data();
  for (Eigen::Index i = 0; i < weights_.data.size(); ++i) {
    w_ptr[i] = dist(gen);
  }

  float* b_ptr = bias_.data.data();
  for (Eigen::Index i = 0; i < bias_.data.size(); ++i) {
    b_ptr[i] = dist(gen);
  }
}

autograd::Tensor* DenseLayer::forward(autograd::Tensor* input) {
  auto* tape = autograd::Tape::get_global();

  bool req_grad_mm =
      tape->record_ops_ && (input->requires_grad || weights_.requires_grad);
  auto* mm = tape->alloc_tensor(
      mlengine::Shape{input->shape[0], weights_.shape[1]}, req_grad_mm);
  auto* mm_op = tape->allocate_op<mlengine::autograd::ops::MatMulOp>(
      input, &weights_, mm);
  mm_op->forward();

  bool req_grad_out =
      tape->record_ops_ && (mm->requires_grad || bias_.requires_grad);
  auto* out = tape->alloc_tensor(mm->shape, req_grad_out);
  auto* bias_op =
      tape->allocate_op<mlengine::autograd::ops::AddBiasOp>(mm, &bias_, out);
  bias_op->forward();

  return out;
}

std::vector<autograd::Tensor*> DenseLayer::parameters() {
  return {&weights_, &bias_};
}

}  // namespace mlengine::layers