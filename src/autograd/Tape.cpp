#include "autograd/Tape.hpp"

#include <algorithm>
#include <stdexcept>

namespace mlengine::autograd {

Tape*& Tape::global_tape() {
  static thread_local Tape* current = nullptr;
  return current;
}

void Tape::set_global(Tape* t) { global_tape() = t; }

Tape* Tape::get_global() {
  Tape* t = global_tape();
  if (!t) throw std::runtime_error("No active tape context.");
  return t;
}

Tape::Tape(bool record_ops) : record_ops_(record_ops) {
  execution_order_.reserve(10000);
  arena_ops_.reserve(10000);
  op_arena_.emplace_back(BLOCK_SIZE);
}

Tape::~Tape() {
  for (Op* op : arena_ops_) {
    op->~Op();
  }
}

Tensor* Tape::alloc_tensor(const mlengine::Shape& shape, bool requires_grad) {
  bool req_grad_actual = record_ops_ && requires_grad;

  if (tensor_idx_ >= tensor_pool_.size()) {
    tensor_pool_.emplace_back(shape, req_grad_actual);
  } else {
    tensor_pool_[tensor_idx_].requires_grad = req_grad_actual;
    tensor_pool_[tensor_idx_].resize(shape);
    if (req_grad_actual) {
      tensor_pool_[tensor_idx_].grad.setZero();
    }
  }
  return &tensor_pool_[tensor_idx_++];
}

Tensor* Tape::push_tensor(const Tensor& input_data, bool requires_grad) {
  Tensor* t = alloc_tensor(input_data.shape, requires_grad);
  std::copy(input_data.data.data(),
            input_data.data.data() + input_data.data.size(), t->data.data());
  return t;
}

void Tape::record_op(std::shared_ptr<Op> op) {
  if (record_ops_) {
    execution_order_.push_back(op.get());
    external_ops_.push_back(op);
  }
}

void Tape::replay_forward() {
  for (auto* op : execution_order_) op->forward();
}

void Tape::replay_backward() {
  for (auto it = execution_order_.rbegin(); it != execution_order_.rend();
       ++it) {
    (*it)->backward();
  }
}

void Tape::zero_grads() {
  for (auto& t : tensor_pool_) t.zero_grad();
}

void Tape::backward() { replay_backward(); }

void Tape::reset() {
  for (Op* op : arena_ops_) {
    op->~Op();
  }
  arena_ops_.clear();

  execution_order_.clear();
  external_ops_.clear();

  current_block_ = 0;
  arena_offset_ = 0;
  tensor_idx_ = 0;
}

TapeGuard::TapeGuard(Tape* new_tape) {
  prev_tape_ = Tape::global_tape();
  Tape::set_global(new_tape);
}

TapeGuard::~TapeGuard() { Tape::set_global(prev_tape_); }

}  // namespace mlengine::autograd