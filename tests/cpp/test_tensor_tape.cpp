#include <gtest/gtest.h>

#include "autograd/Tape.hpp"
#include "autograd/Tensor.hpp"
#include "core/Types.hpp"

using namespace mlengine;
using namespace mlengine::autograd;

// ==============================================================================
// Tensor Memory and Shape Management
// ==============================================================================
class TensorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup code executes before every TEST_F
  }
  void TearDown() override {
    // Cleanup code executes after every TEST_F
  }
};

TEST_F(TensorTest, InitializationAndSizeComputation) {
  Shape shape = {2, 3, 4};
  Tensor t(shape, true, false);  //[cite: 1]

  EXPECT_EQ(t.shape, shape);
  EXPECT_EQ(t.data.size(), 24);
  EXPECT_EQ(t.grad.size(), 24);  // Requires grad is true[cite: 1]
  EXPECT_TRUE(t.requires_grad);
}

TEST_F(TensorTest, ResizingAndZeroingGradients) {
  Tensor t({2, 2});
  t.grad.setConstant(5.0f);

  t.zero_grad();  //[cite: 1]
  for (Eigen::Index i = 0; i < t.grad.size(); ++i) {
    EXPECT_FLOAT_EQ(t.grad[i], 0.0f);
  }

  t.resize({4, 4});  //[cite: 1]
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
  Tape tape1(true);  //[cite: 1]
  {
    TapeGuard guard(&tape1);                //[cite: 1]
    EXPECT_EQ(Tape::get_global(), &tape1);  //[cite: 1]
  }
  // Assuming no previous tape was set in this thread, it should throw
  EXPECT_THROW(Tape::get_global(), std::runtime_error);  //[cite: 1]
}

TEST_F(TapeTest, TensorPoolingAndMemoryReuse) {
  Tape tape(true);
  TapeGuard guard(&tape);

  Tensor* t1 = tape.alloc_tensor({2, 2}, true);   //[cite: 1]
  Tensor* t2 = tape.alloc_tensor({3, 3}, false);  //[cite: 1]

  EXPECT_EQ(tape.tensor_pool_.size(), 2);  //[cite: 1]
  EXPECT_EQ(t1->shape, (Shape{2, 2}));

  // Reset tape (simulating next training epoch)
  tape.reset();                    //[cite: 1]
  EXPECT_EQ(tape.tensor_idx_, 0);  //[cite: 1]

  // Next allocation should reuse the pool rather than reallocating
  Tensor* t3 = tape.alloc_tensor({4, 4}, true);  //[cite: 1]
  EXPECT_EQ(tape.tensor_pool_.size(), 2);  // Pool size shouldn't grow[cite: 1]
  EXPECT_EQ(t3->shape, (Shape{4, 4}));     // But shape should update[cite: 1]
}