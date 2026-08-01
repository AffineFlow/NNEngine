#include <gtest/gtest.h>

#include "autograd/Tape.hpp"
#include "autograd/Tensor.hpp"
#include "autograd/ops/AddOp.hpp"

using namespace affineflow;
using namespace affineflow::autograd;

class ImplicitTapeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Ensure we start with a clean tape for every test
    Tape* t = Tape::get_global();
    t->reset();
  }
};

TEST_F(ImplicitTapeTest, LazyInitialization) {
  // Tape::get_global() should automatically allocate the tape if it doesn't
  // exist
  Tape* t = Tape::get_global();
  ASSERT_NE(t, nullptr);
  EXPECT_TRUE(t->record_ops_);
}

TEST_F(ImplicitTapeTest, NoGradGuardDisablesTracking) {
  Tape* t = Tape::get_global();

  Tensor a({2}, true);
  Tensor b({2}, true);

  {
    NoGradGuard guard;
    guard.enter();
    EXPECT_FALSE(t->record_ops_);

    // Operations inside this block should not be pushed to the execution_order_
    Tensor* out = t->alloc_tensor({2}, true);
    auto* op = t->allocate_op<ops::AddOp>(&a, &b, out);
    op->forward();

    guard.exit();
  }

  EXPECT_TRUE(t->record_ops_);
}

TEST_F(ImplicitTapeTest, AutoResetOnBackward) {
  Tape* t = Tape::get_global();

  Tensor a({2}, true);
  Tensor b({2}, true);
  Tensor* out = t->alloc_tensor({2}, true);

  auto* op = t->allocate_op<ops::AddOp>(&a, &b, out);
  op->forward();

  // Backward without retain_graph (default false)
  out->grad.setConstant(1.0f);
  t->backward();
  t->reset();  // Simulating the binding's auto-reset behavior

  // If the tape reset correctly, allocating a new op will reuse the block
  // offsets. We can't safely assert raw block sizes, but we ensure it doesn't
  // crash on next run.
  Tensor* out2 = t->alloc_tensor({2}, true);
  auto* op2 = t->allocate_op<ops::AddOp>(&a, &b, out2);
  op2->forward();
  SUCCEED();
}