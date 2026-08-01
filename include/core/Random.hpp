#pragma once

#include <cstdint>
#include <cstdlib>
#include <random>

namespace affineflow::core {

/**
 * @brief Return the process-wide Mersenne Twister used for initialization.
 */
inline std::mt19937& rng() {
  static std::mt19937 engine{std::random_device{}()};
  return engine;
}

/**
 * @brief Seed the shared RNGs to make initialization, shuffling, and dropout
 * repeatable.
 * @param seed Seed value to apply to the global generators.
 */
inline void set_seed(std::uint32_t seed) {
  rng().seed(seed);  // Seeds C++11 std::mt19937 (used by layers and DataLoader)
  std::srand(
      seed);  // Seeds C std::rand (used by Eigen::Matrix::Random for Dropout)
}

/**
 * @brief Generate a normally distributed random float.
 */
inline float random_normal(float mean = 0.0f, float stddev = 1.0f) {
  std::normal_distribution<float> dist(mean, stddev);
  return dist(rng());
}

}  // namespace affineflow::core