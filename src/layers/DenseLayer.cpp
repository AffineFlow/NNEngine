#include "layers/DenseLayer.hpp"

#include <cmath>
#include <memory>
#include <random>

#include "autograd/Tape.hpp"
#include "autograd/ops/AddBiasOp.hpp"
#include "autograd/ops/MatMulOp.hpp"
#include "core/Random.hpp"

namespace mlengine::layers {

DenseLayer::DenseLayer(int input_dim, int output_dim)
    : weights_(core::MatrixRM::Zero(input_dim, output_dim), true, true),
      bias_(core::MatrixRM::Zero(1, output_dim), true, false) {
  float limit = std::sqrt(6.0f / static_cast<float>(input_dim + output_dim));
  std::uniform_real_distribution<float> dist(-limit, limit);
  auto& gen = core::rng();

  for (int i = 0; i < weights_.data.rows(); ++i) {
    for (int j = 0; j < weights_.data.cols(); ++j) {
      weights_.data(i, j) = dist(gen);
    }
  }
}

autograd::Tensor* DenseLayer::forward(autograd::Tensor* input) {
  auto* tape = autograd::Tape::get_global();
  bool req_grad_mm =
      tape->record_ops_ && (input->requires_grad || weights_.requires_grad);
  auto* mm =
      tape->alloc_tensor(input->data.rows(), weights_.data.cols(), req_grad_mm);
  auto mm_op = std::make_shared<autograd::ops::MatMulOp>(input, &weights_, mm);
  mm_op->forward();
  tape->record_op(mm_op);

  bool req_grad_out =
      tape->record_ops_ && (mm->requires_grad || bias_.requires_grad);
  auto* out =
      tape->alloc_tensor(mm->data.rows(), bias_.data.cols(), req_grad_out);
  auto bias_op = std::make_shared<autograd::ops::AddBiasOp>(mm, &bias_, out);
  bias_op->forward();
  tape->record_op(bias_op);

  return out;
}

std::vector<autograd::Tensor*> DenseLayer::parameters() {
  return {&weights_, &bias_};
}

}  // namespace mlengine::layers