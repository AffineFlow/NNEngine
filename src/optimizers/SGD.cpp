#include "optimizers/SGD.hpp"

namespace affineflow::nn::core {
SGD::SGD(float learning_rate) : Optimizer(learning_rate) {}
void SGD::step() {
  for (auto* p : parameters_) {
    if (!p->requires_grad) continue;

    float* p_ptr = p->data.data();
    float* g_ptr = p->grad.data();
    Eigen::Index size = p->data.size();

#pragma omp simd
    for (Eigen::Index j = 0; j < size; ++j) {
      p_ptr[j] -= g_ptr[j] * lr_;
    }
  }
}
}  // namespace affineflow::nn::core