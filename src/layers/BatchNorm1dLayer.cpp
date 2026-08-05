#include "layers/BatchNorm1dLayer.hpp"

#include "autograd/Tape.hpp"
#include "autograd/ops/BatchNorm1dOp.hpp"

namespace affineflow::nn::layers {

BatchNorm1dLayer::BatchNorm1dLayer(int num_features, float eps, float momentum)
    : gamma_(affineflow::nn::Shape{1, num_features}, true),
      beta_(affineflow::nn::Shape{1, num_features}, true),
      running_mean_(affineflow::nn::Shape{1, num_features}, false),
      running_var_(affineflow::nn::Shape{1, num_features}, false),
      momentum_(momentum),
      eps_(eps) {
  gamma_.data.setConstant(1.0f);
  running_var_.data.setConstant(1.0f);
}

autograd::Tensor* BatchNorm1dLayer::forward(autograd::Tensor* input) {
  auto* tape = autograd::Tape::get_global();
  bool req_grad =
      tape->record_ops_ &&
      (input->requires_grad || gamma_.requires_grad || beta_.requires_grad);
  auto* out = tape->alloc_tensor(input->shape, req_grad);

  auto* op = tape->allocate_op<affineflow::nn::autograd::ops::BatchNorm1dOp>(
      input, &gamma_, &beta_, out, &running_mean_, &running_var_, momentum_,
      eps_, &is_training_);
  op->forward();

  return out;
}

std::vector<autograd::Tensor*> BatchNorm1dLayer::parameters() {
  return {&gamma_, &beta_};
}

std::map<std::string, autograd::Tensor*> BatchNorm1dLayer::named_parameters() {
  return {{"weight", &gamma_},
          {"bias", &beta_},
          {"running_mean", &running_mean_},
          {"running_var", &running_var_}};
}

}  // namespace affineflow::nn::layers