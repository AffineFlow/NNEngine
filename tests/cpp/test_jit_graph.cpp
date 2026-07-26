#include <gtest/gtest.h>

#include "core/JITGraph.hpp"
#include "layers/DenseLayer.hpp"

using namespace mlengine;

TEST(JITGraphTest, TraceForwardPass) {
  core::JITGraph graph;
  layers::DenseLayer dense(4, 4);

  Tensor input({1, 4});
  input.fill(1.0f);

  // Trace the computation
  graph.begin_trace();
  Tensor output = dense.forward(input);
  graph.end_trace();

  EXPECT_TRUE(graph.is_compiled());
  EXPECT_GT(graph.node_count(), 0);

  // Execute compiled graph and ensure it matches eager mode
  Tensor compiled_output = graph.execute({input});

  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_FLOAT_EQ(output.data()[i], compiled_output.data()[i]);
  }
}