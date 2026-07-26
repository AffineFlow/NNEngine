#pragma once
#include "autograd/Op.hpp"
#include "autograd/Tensor.hpp"

namespace mlengine::autograd::ops {

class AddScalarOp : public Op {
  Tensor *a_, *out_;
  float val_;

 public:
  AddScalarOp(Tensor* a, float val, Tensor* out)
      : a_(a), val_(val), out_(out) {}
  void forward() override;
  void backward() override;
};

class SubScalarOp : public Op {
  Tensor *a_, *out_;
  float val_;

 public:
  SubScalarOp(Tensor* a, float val, Tensor* out)
      : a_(a), val_(val), out_(out) {}
  void forward() override;
  void backward() override;
};

class RSubScalarOp : public Op {
  Tensor *a_, *out_;
  float val_;

 public:
  RSubScalarOp(Tensor* a, float val, Tensor* out)
      : a_(a), val_(val), out_(out) {}
  void forward() override;
  void backward() override;
};

class MulScalarOp : public Op {
  Tensor *a_, *out_;
  float val_;

 public:
  MulScalarOp(Tensor* a, float val, Tensor* out)
      : a_(a), val_(val), out_(out) {}
  void forward() override;
  void backward() override;
};

class DivScalarOp : public Op {
  Tensor *a_, *out_;
  float val_;

 public:
  DivScalarOp(Tensor* a, float val, Tensor* out)
      : a_(a), val_(val), out_(out) {}
  void forward() override;
  void backward() override;
};

class RDivScalarOp : public Op {
  Tensor *a_, *out_;
  float val_;

 public:
  RDivScalarOp(Tensor* a, float val, Tensor* out)
      : a_(a), val_(val), out_(out) {}
  void forward() override;
  void backward() override;
};

}  // namespace mlengine::autograd::ops