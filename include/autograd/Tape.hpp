#pragma once
#include <deque>
#include <memory>
#include <vector>

#include "autograd/Op.hpp"
#include "autograd/Tensor.hpp"

namespace mlengine::autograd {

class Tape {
 public:
  std::deque<Tensor> tensor_pool_;
  size_t tensor_idx_ = 0;
  std::vector<std::shared_ptr<Op>> ops_;
  bool record_ops_;

  static Tape*& global_tape();
  static void set_global(Tape* t);
  static Tape* get_global();

  explicit Tape(bool record_ops = true);

  Tensor* alloc_tensor(const mlengine::Shape& shape, bool requires_grad = true);

  // Bridge for MatrixRM inputs
  Tensor* push_tensor(const mlengine::MatrixRM& data,
                      bool requires_grad = true);

  void record_op(std::shared_ptr<Op> op);
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

}  // namespace mlengine::autograd