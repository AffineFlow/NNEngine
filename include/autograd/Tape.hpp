#pragma once
#include <algorithm>
#include <cstddef>
#include <deque>
#include <memory>
#include <vector>

#include "autograd/Op.hpp"
#include "autograd/Tensor.hpp"

namespace affineengine::autograd {

class Tape {
 private:
  std::vector<std::vector<char>> op_arena_;
  size_t current_block_ = 0;
  size_t arena_offset_ = 0;
  const size_t BLOCK_SIZE = 1024 * 1024;  // 1MB blocks

  std::vector<Op*>
      execution_order_;         // Ordered list for forward/backward replay
  std::vector<Op*> arena_ops_;  // C++ Ops requiring manual destruction
  std::vector<std::shared_ptr<Op>>
      external_ops_;  // Python Ops requiring shared_ptr lifecycle

 public:
  std::deque<Tensor> tensor_pool_;
  size_t tensor_idx_ = 0;
  bool record_ops_;

  static Tape*& global_tape();
  static void set_global(Tape* t);
  static Tape* get_global();

  explicit Tape(bool record_ops = true);
  ~Tape();

  Tensor* alloc_tensor(const affineengine::Shape& shape, bool requires_grad = true);
  Tensor* push_tensor(const Tensor& data, bool requires_grad = true);

  void record_op(std::shared_ptr<Op> op);

  template <typename T, typename... Args>
  T* allocate_op(Args&&... args) {
    size_t alignment = alignof(T);
    arena_offset_ = (arena_offset_ + alignment - 1) & ~(alignment - 1);

    if (op_arena_.empty() ||
        arena_offset_ + sizeof(T) > op_arena_[current_block_].size()) {
      if (current_block_ + 1 < op_arena_.size() && !op_arena_.empty()) {
        current_block_++;
      } else {
        op_arena_.emplace_back(std::max(BLOCK_SIZE, sizeof(T) + alignment));
        current_block_ = op_arena_.size() - 1;
      }
      arena_offset_ = 0;
      arena_offset_ = (arena_offset_ + alignment - 1) & ~(alignment - 1);
    }

    T* op_ptr = new (op_arena_[current_block_].data() + arena_offset_)
        T(std::forward<Args>(args)...);
    arena_offset_ += sizeof(T);

    if (record_ops_) {
      execution_order_.push_back(op_ptr);
      arena_ops_.push_back(op_ptr);
    }
    return op_ptr;
  }

  void replay_forward();
  void replay_backward();
  void zero_grads();
  void backward();
  void reset();
};

class TapeGuard {
  Tape* prev_tape_;

 public:
  explicit TapeGuard(Tape* new_tape);
  ~TapeGuard();
};

}  // namespace affineengine::autograd