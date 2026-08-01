#include <gtest/gtest.h>

#include "autograd/Tape.hpp"
#include "autograd/Tensor.hpp"
#include "autograd/ops/AddBiasOp.hpp"
#include "autograd/ops/AddOp.hpp"
#include "autograd/ops/MulOp.hpp"

using namespace affineflow;
using namespace affineflow::autograd;

TEST(AutogradTest, TapeContextManagement) {
  // Implicitly initializes
  EXPECT_NE(Tape::get_global(), nullptr);

  {
    Tape tape(true);
    TapeGuard guard(&tape);
    EXPECT_EQ(Tape::get_global(), &tape);
  }

  // Returns to the implicit thread-local tape
  EXPECT_NE(Tape::get_global(), nullptr);
}

TEST(AutogradTest, TapeNodeRecording) {
  Tape tape(true);
  TapeGuard guard(&tape);

  auto* t1 = tape.push_tensor(Tensor({2, 2}), true);
  auto* t2 = tape.push_tensor(Tensor({2, 2}), true);
  auto* t3 = tape.push_tensor(Tensor({2, 2}), true);

  auto* op = tape.allocate_op<ops::AddBiasOp>(t1, t2, t3);
  op->forward();

  EXPECT_TRUE(t1->requires_grad);
}

TEST(AutogradTest, MathOpsExecution) {
  Tape tape(true);
  TapeGuard guard(&tape);

  auto* t1 = tape.push_tensor(Tensor({2}), true);
  auto* t2 = tape.push_tensor(Tensor({2}), true);

  t1->data.setConstant(2.0f);
  t2->data.setConstant(3.0f);

  auto* t3 = tape.alloc_tensor({2}, true);
  auto* add_op = tape.allocate_op<ops::AddOp>(t1, t2, t3);
  add_op->forward();

  EXPECT_FLOAT_EQ(t3->data.data()[0], 5.0f);

  t3->grad.setConstant(1.0f);
  tape.backward();

  EXPECT_FLOAT_EQ(t1->grad.data()[0], 1.0f);
  EXPECT_FLOAT_EQ(t2->grad.data()[0], 1.0f);
}