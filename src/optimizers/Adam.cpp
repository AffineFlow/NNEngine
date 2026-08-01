#include "optimizers/Adam.hpp"

#include <cmath>

namespace affineengine::core {

Adam::Adam(float learning_rate) : Optimizer(learning_rate) {}

void Adam::set_parameters(const std::vector<autograd::Tensor*>& params) {
  Optimizer::set_parameters(params);

  // Only initialize if the momentum vectors don't exist yet
  if (m_.size() == parameters_.size()) return;

  m_.clear();
  v_.clear();
  t_ = 0;
  for (auto* p : parameters_) {
    m_.push_back(affineengine::FlatStorage(p->data.size()));
    m_.back().setZero();
    v_.push_back(affineengine::FlatStorage(p->data.size()));
    v_.back().setZero();
  }
}

void Adam::step() {
  t_++;
  float current_lr = lr_;
  float bias_corr1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
  float bias_corr2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));

  for (size_t i = 0; i < parameters_.size(); ++i) {
    auto* p = parameters_[i];
    if (!p->requires_grad) continue;

    float* p_ptr = p->data.data();
    float* g_ptr = p->grad.data();
    float* m_ptr = m_[i].data();
    float* v_ptr = v_[i].data();
    Eigen::Index size = p->data.size();

#pragma omp simd
    for (Eigen::Index j = 0; j < size; ++j) {
      m_ptr[j] = beta1_ * m_ptr[j] + (1.0f - beta1_) * g_ptr[j];
      v_ptr[j] = beta2_ * v_ptr[j] + (1.0f - beta2_) * (g_ptr[j] * g_ptr[j]);

      p_ptr[j] -= current_lr * (m_ptr[j] / bias_corr1) /
                  (std::sqrt(v_ptr[j] / bias_corr2) + epsilon_);
    }
  }
}

void Adam::save_state(std::ostream& os) const {
  os.write(reinterpret_cast<const char*>(&t_), sizeof(int));
  size_t num_params = parameters_.size();
  os.write(reinterpret_cast<const char*>(&num_params), sizeof(size_t));

  for (const auto& mat : m_) {
    size_t size = mat.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
    os.write(reinterpret_cast<const char*>(mat.data()), size * sizeof(float));
  }
  for (const auto& mat : v_) {
    os.write(reinterpret_cast<const char*>(mat.data()),
             mat.size() * sizeof(float));
  }
}

void Adam::load_state(std::istream& is) {
  is.read(reinterpret_cast<char*>(&t_), sizeof(int));
  size_t num_params;
  is.read(reinterpret_cast<char*>(&num_params), sizeof(size_t));

  m_.resize(num_params);
  v_.resize(num_params);

  for (size_t i = 0; i < num_params; ++i) {
    size_t size;
    is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    m_[i].resize(size);
    is.read(reinterpret_cast<char*>(m_[i].data()), size * sizeof(float));
  }
  for (size_t i = 0; i < num_params; ++i) {
    size_t size = m_[i].size();
    v_[i].resize(size);
    is.read(reinterpret_cast<char*>(v_[i].data()), size * sizeof(float));
  }
}
}  // namespace affineengine::core