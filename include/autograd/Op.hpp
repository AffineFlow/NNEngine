#pragma once

namespace affineflow::nn::autograd {

/**
 * @brief Base interface for a differentiable primitive recorded on the tape.
 *
 * Ops encapsulate a forward computation together with its adjoint so the tape
 * can replay the graph without heap allocations in the training loop.
 */
class Op {
 public:
  virtual ~Op() = default;
  /**
   * @brief Execute the forward pass for the primitive.
   */
  virtual void forward() = 0;
  /**
   * @brief Accumulate gradients for the primitive inputs.
   */
  virtual void backward() = 0;
};

}  // namespace affineflow::nn::autograd