#include <gtest/gtest.h>

#include "autograd/Op.hpp"
#include "autograd/Tape.hpp"
#include "autograd/Tensor.hpp"

using namespace affineengine;

// Dummy Op to track allocations and destructions in the Arena
static int dummy_op_destructions = 0;

class DummyArenaOp : public autograd::Op {
 public:
  DummyArenaOp() {}
  ~DummyArenaOp() override { dummy_op_destructions++; }
  void forward() override {}
  void backward() override {}
};

// Dummy Python Op to ensure shared_ptr instances are not double-deleted
class DummyPythonOp : public autograd::Op {
 public:
  DummyPythonOp() {}
  ~DummyPythonOp() override {}
  void forward() override {}
  void backward() override {}
};

TEST(EagerTapeTest, ArenaBlockAllocationAndDestruction) {
  dummy_op_destructions = 0;

  {
    autograd::Tape tape(true);
    autograd::TapeGuard guard(&tape);

    // Allocate 100 dummy operations dynamically
    for (int i = 0; i < 100; ++i) {
      tape.allocate_op<DummyArenaOp>();
    }

    // Resetting should manually destruct the 100 Ops
    tape.reset();
    EXPECT_EQ(dummy_op_destructions, 100);

    // Second pass to ensure block reuse doesn't crash and memory aligns
    for (int i = 0; i < 50; ++i) {
      tape.allocate_op<DummyArenaOp>();
    }
    tape.reset();
    EXPECT_EQ(dummy_op_destructions, 150);
  }
}

TEST(EagerTapeTest, MixedPythonAndNativeOps) {
  dummy_op_destructions = 0;  // Reset global state for isolated CTest runs

  autograd::Tape tape(true);
  autograd::TapeGuard guard(&tape);

  // Allocate Native Op
  tape.allocate_op<DummyArenaOp>();

  // Allocate "Python" Op (simulating pybind11 shared_ptr lifecycle)
  auto py_op = std::make_shared<DummyPythonOp>();
  tape.record_op(
      py_op);  // External ops should be tracked but not manually destructed

  // If the tape incorrectly deletes the shared_ptr op, this will segfault
  tape.reset();

  // Verify native op was destructed successfully
  EXPECT_EQ(dummy_op_destructions,
            1);  // Expect exactly 1 destruction in this test
}