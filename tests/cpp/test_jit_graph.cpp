#include <gtest/gtest.h>

#include <memory>

#include "autograd/Tensor.hpp"
#include "core/DataLoader.hpp"
#include "core/JITGraph.hpp"
#include "layers/DenseLayer.hpp"
#include "losses/MSELoss.hpp"
#include "optimizers/SGD.hpp"

using namespace mlengine;
using namespace mlengine::autograd;

TEST(JITGraphTest, TraceAndExecuteBatch) {
  auto dense = std::make_shared<layers::DenseLayer>(4, 4);
  auto opt = std::make_shared<core::SGD>(0.01f);
  auto loss = std::make_shared<core::MSELoss>();

  core::JITGraph graph(dense, opt, loss);

  Tensor input({2, 4});
  input.data.setConstant(1.0f);

  Tensor target({2, 4});
  target.data.setConstant(0.0f);

  core::DataLoader loader(input, target, 2);

  // Trace the computation (which compiles the graph and does one
  // forward/backward pass)
  float initial_loss = graph.trace_batch(loader);

  EXPECT_GT(initial_loss, 0.0f);

  // Reset loader and run compiled fast loop for the epoch
  loader.reset();
  auto [total_loss, batches] = graph.fast_loop(loader);

  EXPECT_EQ(batches, 1);
}