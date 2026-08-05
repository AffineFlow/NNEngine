#include <gtest/gtest.h>

#include "autograd/Tape.hpp"
#include "autograd/Tensor.hpp"
#include "core/Types.hpp"

using namespace affineflow::nn;
using namespace affineflow::nn::autograd;

// ==============================================================================
// Tensor Memory and Shape Management
// ==============================================================================
class TensorTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(TensorTest, InitializationAndSizeComputation) {
  Shape shape = {2, 3, 4};
  Tensor t(shape, true, false);

  EXPECT_EQ(t.shape, shape);
  EXPECT_EQ(t.data.size(), 24);
  EXPECT_EQ(t.grad.size(), 24);
  EXPECT_TRUE(t.requires_grad);
}

TEST_F(TensorTest, ResizingAndZeroingGradients) {
  Tensor t({2, 2});
  t.grad.setConstant(5.0f);

  t.zero_grad();
  for (Eigen::Index i = 0; i < t.grad.size(); ++i) {
    EXPECT_FLOAT_EQ(t.grad[i], 0.0f);
  }

  t.resize({4, 4});
  EXPECT_EQ(t.shape, (Shape{4, 4}));
  EXPECT_EQ(t.data.size(), 16);
}

// ==============================================================================
// Tape Context and Allocation
// ==============================================================================
class TapeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(TapeTest, GlobalTapeGuardScoping) {
  Tape tape1(true);
  {
    TapeGuard guard(&tape1);
    EXPECT_EQ(Tape::get_global(), &tape1);
  }
  // The global tape is now implicitly lazy-initialized, so it shouldn't throw.
  EXPECT_NE(Tape::get_global(), nullptr);
}

TEST_F(TapeTest, TensorPoolingAndMemoryReuse) {
  Tape tape(true);
  TapeGuard guard(&tape);

  Tensor* t1 = tape.alloc_tensor({2, 2}, true);
  Tensor* t2 = tape.alloc_tensor({3, 3}, false);

  EXPECT_EQ(tape.tensor_pool_.size(), 2);
  EXPECT_EQ(t1->shape, (Shape{2, 2}));

  // Reset tape (simulating next training epoch)
  tape.reset();
  EXPECT_EQ(tape.tensor_idx_, 0);

  // Next allocation should reuse the pool rather than reallocating
  Tensor* t3 = tape.alloc_tensor({4, 4}, true);
  EXPECT_EQ(tape.tensor_pool_.size(), 2);  // Pool size shouldn't grow
  EXPECT_EQ(t3->shape, (Shape{4, 4}));     // But shape should update
}