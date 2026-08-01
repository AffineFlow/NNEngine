#pragma once
#include <Eigen/Core>
#include <numeric>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace affineflow {

using Scalar = float;

// Flat 1D storage for N-dimensional tensor data
using FlatStorage = Eigen::Tensor<Scalar, 1, Eigen::RowMajor>;

// Dynamic shape representation
using Shape = std::vector<Eigen::Index>;

// Helper to calculate total elements from a shape
inline Eigen::Index compute_size(const Shape& shape) {
  if (shape.empty()) return 0;
  return std::accumulate(shape.begin(), shape.end(), 1,
                         std::multiplies<Eigen::Index>());
}

}  // namespace affineflow