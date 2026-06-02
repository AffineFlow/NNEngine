#include "optimizers/Adam.hpp"

#include <cmath>

namespace mlengine::core {

Adam::Adam(float learning_rate) : Optimizer(learning_rate) {}

void Adam::set_parameters(const std::vector<autograd::Tensor*>& params) {
  Optimizer::set_parameters(params);
  m_.clear();
  v_.clear();
  t_ = 0;
  for (auto* p : parameters_) {
    m_.push_back(autograd::MatrixRM::Zero(p->data.rows(), p->data.cols()));
    v_.push_back(autograd::MatrixRM::Zero(p->data.rows(), p->data.cols()));
  }
}

void Adam::step() {
  t_++;
  float current_lr = lr_;

  double bias_corr1 =
      1.0 - std::pow(static_cast<double>(beta1_), static_cast<double>(t_));
  double bias_corr2 =
      1.0 - std::pow(static_cast<double>(beta2_), static_cast<double>(t_));

  for (size_t i = 0; i < parameters_.size(); ++i) {
    auto* p = parameters_[i];
    if (!p->requires_grad) continue;

    float* p_ptr = p->data.data();
    float* g_ptr = p->grad.data();
    float* m_ptr = m_[i].data();
    float* v_ptr = v_[i].data();
    size_t size = p->data.size();

#pragma omp simd
    for (size_t j = 0; j < size; ++j) {
      m_ptr[j] = beta1_ * m_ptr[j] + (1.0f - beta1_) * g_ptr[j];
      v_ptr[j] = beta2_ * v_ptr[j] + (1.0f - beta2_) * (g_ptr[j] * g_ptr[j]);

      p_ptr[j] -=
          current_lr * (m_ptr[j] / static_cast<float>(bias_corr1)) /
          (std::sqrt(v_ptr[j] / static_cast<float>(bias_corr2)) + epsilon_);
    }
  }
}

void Adam::save_state(std::ostream& os) const {
  os.write(reinterpret_cast<const char*>(&t_), sizeof(int));
  size_t num_params = parameters_.size();
  os.write(reinterpret_cast<const char*>(&num_params), sizeof(size_t));

  for (const auto& mat : m_) {
    size_t rows = mat.rows(), cols = mat.cols();
    os.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    os.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    os.write(reinterpret_cast<const char*>(mat.data()),
             rows * cols * sizeof(float));
  }
  for (const auto& mat : v_) {
    // Both moments share identical shapes
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
    size_t rows, cols;
    is.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
    is.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
    m_[i].resize(rows, cols);
    is.read(reinterpret_cast<char*>(m_[i].data()), rows * cols * sizeof(float));
  }
  for (size_t i = 0; i < num_params; ++i) {
    size_t rows = m_[i].rows(), cols = m_[i].cols();
    v_[i].resize(rows, cols);
    is.read(reinterpret_cast<char*>(v_[i].data()), rows * cols * sizeof(float));
  }
}

}  // namespace mlengine::core
