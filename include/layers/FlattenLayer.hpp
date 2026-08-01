#pragma once
#include "core/Layer.hpp"

namespace affineflow::layers {

class FlattenLayer : public core::Layer {
 public:
  autograd::Tensor* forward(autograd::Tensor* input) override;
};

}  // namespace affineflow::layers