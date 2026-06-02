#pragma once

#include <deque>
#include <memory>
#include <vector>

#include "autograd/Op.hpp"
#include "autograd/Tensor.hpp"

namespace mlengine::autograd {

/**
 * @brief Arena-style memory allocator and operation replay log.
 */
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

  Tensor* alloc_tensor(int rows, int cols, bool requires_grad = true);

  /** @brief Template implementation for zero-copy block passing. Must reside in
   * header. */
  template <typename Derived>
  Tensor* push_expr(const Eigen::MatrixBase<Derived>& expr,
                    bool requires_grad = true) {
    Tensor* t = alloc_tensor(expr.rows(), expr.cols(), requires_grad);
    t->data.noalias() = expr;
    return t;
  }

  Tensor* push_tensor(const mlengine::MatrixRM& data,
                      bool requires_grad = true);
  void record_op(std::shared_ptr<Op> op);
  void replay_forward();
  void replay_backward();
  void zero_grads();
  void backward();
  void reset();
};

/**
 * @brief Thread-safe RAII Guard for managing the active tape context.
 */
class TapeGuard {
  Tape* prev_tape_;

 public:
  explicit TapeGuard(Tape* new_tape);
  ~TapeGuard();
};

}  // namespace mlengine::autograd