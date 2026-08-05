#pragma once
#include "core/Layer.hpp"

namespace affineflow::nn::layers {

class FlattenLayer : public core::Layer {
 public:
  autograd::Tensor* forward(autograd::Tensor* input) override;
};

}  // namespace affineflow::nn::layers