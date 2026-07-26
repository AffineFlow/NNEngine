#include <gtest/gtest.h>

#include "autograd/Tape.hpp"
#include "autograd/Tensor.hpp"
#include "autograd/ops/AddBiasOp.hpp"
#include "autograd/ops/AddOp.hpp"
#include "autograd/ops/MulOp.hpp"

using namespace mlengine;

TEST(AutogradTest, TapeContextManagement) {
  EXPECT_THROW(autograd::Tape::get_global(), std::runtime_error);

  {
    autograd::Tape tape(true);
    autograd::TapeGuard guard(&tape);
    EXPECT_EQ(autograd::Tape::get_global(), &tape);
  }

  EXPECT_THROW(autograd::Tape::get_global(), std::runtime_error);
}

TEST(AutogradTest, TapeNodeRecording) {
  autograd::Tape tape(true);
  autograd::TapeGuard guard(&tape);

  auto t1 = tape.push_tensor(autograd::Tensor({2, 2}), true);
  auto t2 = tape.push_tensor(autograd::Tensor({2, 2}), true);
  auto t3 = tape.push_tensor(autograd::Tensor({2, 2}), true);

  auto op = std::make_shared<autograd::ops::AddBiasOp>(t1, t2, t3);
  tape.record_op(op);

  EXPECT_TRUE(t1->requires_grad);
  EXPECT_EQ(tape.ops_.size(), 1);
}

TEST(AutogradTest, MathOpsExecution) {
  autograd::Tape tape(true);
  autograd::TapeGuard guard(&tape);

  auto t1 = tape.push_tensor(autograd::Tensor({2}), true);
  auto t2 = tape.push_tensor(autograd::Tensor({2}), true);

  t1->data.setConstant(2.0f);
  t2->data.setConstant(3.0f);

  auto t3 = tape.alloc_tensor({2}, true);
  auto add_op = std::make_shared<autograd::ops::AddOp>(t1, t2, t3);
  add_op->forward();
  tape.record_op(add_op);

  EXPECT_FLOAT_EQ(t3->data[0], 5.0f);

  t3->grad.setConstant(1.0f);
  tape.backward();

  EXPECT_FLOAT_EQ(t1->grad[0], 1.0f);
  EXPECT_FLOAT_EQ(t2->grad[0], 1.0f);
}