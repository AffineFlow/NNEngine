#include <gtest/gtest.h>

#include "autograd/Tape.hpp"
#include "autograd/Tensor.hpp"
#include "autograd/ops/AddBiasOp.hpp"

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