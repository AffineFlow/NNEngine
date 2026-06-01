#pragma once

#include <deque>
#include <memory>
#include <vector>

#include "autograd/Op.hpp"
#include "autograd/Tensor.hpp"

namespace mlengine::autograd {

/**
 * @brief Arena-style storage and replay log for autograd tensors and ops.
 *
 * The tape owns all transient tensors it allocates. Callers receive raw
 * pointers for convenience, but the lifetime is tied to the tape and must not
 * be managed manually.
 */
class Tape {
 public:
  std::deque<Tensor> tensor_pool_;
  size_t tensor_idx_ = 0;
  std::vector<std::shared_ptr<Op>> ops_;
  bool record_ops_;

  /**
   * @brief Thread-local storage for the globally active tape.
   * @return Reference to the current thread's active tape pointer.
   */
  static Tape*& global_tape() {
    static thread_local Tape* current = nullptr;
    return current;
  }

  /**
   * @brief Set the currently active tape for this thread.
   * @param t Pointer to the tape to activate.
   */
  static void set_global(Tape* t) { global_tape() = t; }

  /**
   * @brief Retrieve the currently active tape.
   * @throws std::runtime_error if no tape is currently active.
   * @return Pointer to the active tape.
   */
  static Tape* get_global() {
    Tape* t = global_tape();
    if (!t) {
      throw std::runtime_error(
          "No active tape! Ensure forward() is called within a JITGraph or "
          "predict() context.");
    }
    return t;
  }

  /**
   * @brief Construct a tape that optionally records ops for replay.
   * @param record_ops Whether ops should be stored for backward replay.
   */
  explicit Tape(bool record_ops = true) : record_ops_(record_ops) {
    ops_.reserve(10000);
  }

  /**
   * @brief Allocate a tensor from the tape's arena.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param requires_grad Whether the tensor participates in autograd.
   * @return Pointer to the tape-owned tensor.
   * @note The returned tensor is arena-allocated and owned by the tape, not
   *     the caller.
   */
  Tensor* alloc_tensor(int rows, int cols, bool requires_grad = true) {
    bool req_grad_actual = record_ops_ && requires_grad;
    if (tensor_idx_ >= tensor_pool_.size()) {
      tensor_pool_.emplace_back(mlengine::MatrixRM(rows, cols),
                                req_grad_actual);
    } else {
      if (tensor_pool_[tensor_idx_].data.rows() != rows ||
          tensor_pool_[tensor_idx_].data.cols() != cols) {
        tensor_pool_[tensor_idx_].data.resize(rows, cols);
      }
      tensor_pool_[tensor_idx_].requires_grad = req_grad_actual;
      if (req_grad_actual) {
        if (tensor_pool_[tensor_idx_].grad.rows() != rows ||
            tensor_pool_[tensor_idx_].grad.cols() != cols) {
          tensor_pool_[tensor_idx_].grad.resize(rows, cols);
        }
        tensor_pool_[tensor_idx_].grad.setZero();
      }
    }
    return &tensor_pool_[tensor_idx_++];
  }

  /**
   * @brief Materialize an Eigen expression into tape-owned storage.
   * @tparam Derived Eigen expression type.
   * @param expr Dense expression to copy into a tape tensor.
   * @param requires_grad Whether the resulting tensor requires gradients.
   * @return Pointer to the tape-owned tensor.
   * @note The returned tensor is arena-allocated and owned by the tape, not
   *     the caller.
   */
  template <typename Derived>
  Tensor* push_expr(const Eigen::MatrixBase<Derived>& expr,
                    bool requires_grad = true) {
    Tensor* t = alloc_tensor(expr.rows(), expr.cols(), requires_grad);
    t->data.noalias() = expr;
    return t;
  }

  /**
   * @brief Copy a dense matrix into tape-owned storage.
   * @param data Matrix value to push.
   * @param requires_grad Whether the resulting tensor requires gradients.
   * @return Pointer to the tape-owned tensor.
   * @note The returned tensor is arena-allocated and owned by the tape, not
   *     the caller.
   */
  Tensor* push_tensor(const mlengine::MatrixRM& data,
                      bool requires_grad = true) {
    return push_expr(data, requires_grad);
  }

  /**
   * @brief Record an op for future replay if recording is enabled.
   * @param op Differentiable primitive to append to the tape.
   */
  void record_op(std::shared_ptr<Op> op) {
    if (record_ops_) ops_.push_back(op);
  }

  /**
   * @brief Re-run all recorded forward ops in insertion order.
   */
  void replay_forward() {
    for (auto& op : ops_) op->forward();
  }

  /**
   * @brief Re-run all recorded backward ops in reverse order.
   */
  void replay_backward() {
    for (auto it = ops_.rbegin(); it != ops_.rend(); ++it) (*it)->backward();
  }

  /**
   * @brief Zero every gradient buffer stored in the tape arena.
   */
  void zero_grads() {
    for (auto& t : tensor_pool_) t.zero_grad();
  }

  /**
   * @brief Alias for replaying the backward pass.
   */
  void backward() { replay_backward(); }

  /**
   * @brief Clear the recorded graph and rewind arena allocation.
   */
  void reset() {
    ops_.clear();
    tensor_idx_ = 0;
  }
};

/**
 * @brief RAII Guard to safely manage the thread-local global tape context.
 * Guarantees the tape is cleared even if an exception is thrown during
 * execution.
 */
class TapeGuard {
  Tape* prev_tape_;

 public:
  explicit TapeGuard(Tape* new_tape) {
    prev_tape_ = Tape::global_tape();
    Tape::set_global(new_tape);
  }
  ~TapeGuard() { Tape::set_global(prev_tape_); }
};

}  // namespace mlengine::autograd