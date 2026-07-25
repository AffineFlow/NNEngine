#include "regularizers/L2Regularizer.hpp"

namespace mlengine::core {
L2Regularizer::L2Regularizer(float l2) : l2_(l2) {}

float L2Regularizer::apply(const std::vector<autograd::Tensor*>& parameters) {
  float penalty = 0.0f;
  for (auto* p : parameters) {
    if (!p->requires_grad || !p->apply_regularization) continue;

    float* p_ptr = p->data.data();
    float* g_ptr = p->grad.data();
    Eigen::Index size = p->data.size();
    float local_penalty = 0.0f;

#pragma omp simd reduction(+ : local_penalty)
    for (Eigen::Index j = 0; j < size; ++j) {
      local_penalty += p_ptr[j] * p_ptr[j];
      g_ptr[j] += l2_ * p_ptr[j];
    }
    penalty += 0.5f * l2_ * local_penalty;
  }
  return penalty;
}
}  // namespace mlengine::core