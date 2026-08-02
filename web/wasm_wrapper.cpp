#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <vector>

#include "autograd/Tape.hpp"
#include "autograd/Tensor.hpp"
#include "core/Module.hpp"
#include "layers/Conv2dLayer.hpp"
#include "layers/DenseLayer.hpp"
#include "layers/FlattenLayer.hpp"
#include "layers/LeakyReLULayer.hpp"
#include "layers/Pool2dLayer.hpp"

using namespace emscripten;
using namespace affineflow;

// Replicate the CNN architecture natively
class NNEngineDeepCNN : public core::Module {
  std::shared_ptr<layers::Conv2dLayer> conv1, conv2;
  std::shared_ptr<layers::LeakyReLULayer> act1, act2;
  std::shared_ptr<layers::MaxPool2dLayer> pool1, pool2;
  std::shared_ptr<layers::FlattenLayer> flatten;
  std::shared_ptr<layers::DenseLayer> fc;

 public:
  NNEngineDeepCNN(int in_h, int in_w, int num_classes) {
    conv1 = std::dynamic_pointer_cast<layers::Conv2dLayer>(register_module(
        "conv1",
        std::make_shared<layers::Conv2dLayer>(1, 16, in_h, in_w, 5, 1, 2)));
    act1 = std::dynamic_pointer_cast<layers::LeakyReLULayer>(register_module(
        "act1", std::make_shared<layers::LeakyReLULayer>(0.01f)));
    pool1 = std::dynamic_pointer_cast<layers::MaxPool2dLayer>(register_module(
        "pool1",
        std::make_shared<layers::MaxPool2dLayer>(16, in_h, in_w, 2, 2, 0)));

    int out_h1 = in_h / 2;
    int out_w1 = in_w / 2;
    conv2 = std::dynamic_pointer_cast<layers::Conv2dLayer>(
        register_module("conv2", std::make_shared<layers::Conv2dLayer>(
                                     16, 32, out_h1, out_w1, 3, 1, 1)));
    act2 = std::dynamic_pointer_cast<layers::LeakyReLULayer>(register_module(
        "act2", std::make_shared<layers::LeakyReLULayer>(0.01f)));
    pool2 = std::dynamic_pointer_cast<layers::MaxPool2dLayer>(register_module(
        "pool2",
        std::make_shared<layers::MaxPool2dLayer>(32, out_h1, out_w1, 2, 2, 0)));

    int out_h2 = out_h1 / 2;
    int out_w2 = out_w1 / 2;
    flatten = std::dynamic_pointer_cast<layers::FlattenLayer>(
        register_module("flatten", std::make_shared<layers::FlattenLayer>()));
    fc = std::dynamic_pointer_cast<layers::DenseLayer>(
        register_module("fc", std::make_shared<layers::DenseLayer>(
                                  32 * out_h2 * out_w2, num_classes)));
  }

  autograd::Tensor* forward(autograd::Tensor* x) override {
    x = conv1->forward(x);
    x = act1->forward(x);
    x = pool1->forward(x);
    x = conv2->forward(x);
    x = act2->forward(x);
    x = pool2->forward(x);
    x = flatten->forward(x);
    return fc->forward(x);
  }
};

// ==========================================
// 1. Face Emotion Classifier (64x64 Input)
// ==========================================
class WasmClassifier {
 private:
  int num_classes;
  NNEngineDeepCNN model;

  std::shared_ptr<autograd::Tape> tape;
  autograd::Tensor* X_input;
  autograd::Tensor* predictions;

  std::vector<float> probabilities;

 public:
  WasmClassifier(int classes)
      : num_classes(classes),
        model(64, 64, classes),
        probabilities(classes, 0.0f) {
    try {
      model.load_weights("emotion_weights.nne");
      printf(
          "[Emotion] Loaded trained weights from emotion_weights.nne "
          "successfully!\n");
    } catch (...) {
      printf("[Emotion] Warning: Could not load emotion_weights.nne.\n");
    }

    printf("[Emotion] 1. Initializing Tape...\n");
    model.train(false);
    tape = std::make_shared<autograd::Tape>(true);
    autograd::TapeGuard guard(tape.get());

    printf("[Emotion] 2. Pushing dummy input tensor...\n");
    autograd::Tensor dummy_input({1, 1, 64, 64}, false);
    X_input = tape->push_tensor(dummy_input, false);

    printf("[Emotion] 3. Tracing forward pass (Graph Compilation)...\n");
    predictions = model.forward(X_input);
    printf("[Emotion] 4. Initialization complete!\n");
  }

  val predict(val input_js_array) {
    try {
      std::vector<float> input_data = vecFromJSArray<float>(input_js_array);
      std::copy(input_data.begin(), input_data.end(), X_input->data.data());
      tape->replay_forward();

      float max_val = -std::numeric_limits<float>::infinity();
      for (int i = 0; i < num_classes; ++i) {
        if (predictions->data.data()[i] > max_val)
          max_val = predictions->data.data()[i];
      }

      float sum = 0.0f;
      for (int i = 0; i < num_classes; ++i) {
        probabilities[i] = std::exp(predictions->data.data()[i] - max_val);
        sum += probabilities[i];
      }
      for (int i = 0; i < num_classes; ++i) {
        probabilities[i] /= sum;
      }

      return val::array(probabilities);
    } catch (const std::exception& e) {
      printf("[Emotion] C++ Exception in predict: %s\n", e.what());
      return val::null();
    } catch (...) {
      printf("[Emotion] Unknown C++ Exception in predict\n");
      return val::null();
    }
  }
};

// ==========================================
// 2. MNIST Digit Classifier (28x28 Input)
// ==========================================
class WasmDigitClassifier {
 private:
  int num_classes;
  NNEngineDeepCNN model;

  std::shared_ptr<autograd::Tape> tape;
  autograd::Tensor* X_input;
  autograd::Tensor* predictions;

  std::vector<float> probabilities;

 public:
  WasmDigitClassifier()
      : num_classes(10), model(28, 28, 10), probabilities(10, 0.0f) {
    try {
      model.load_weights("digit_weights.nne");
      printf(
          "[Digit] Loaded trained weights from digit_weights.nne "
          "successfully!\n");
    } catch (...) {
      printf("[Digit] Warning: Could not load digit_weights.nne.\n");
    }

    printf("[Digit] 1. Initializing Tape...\n");
    model.train(false);
    tape = std::make_shared<autograd::Tape>(true);
    autograd::TapeGuard guard(tape.get());

    printf("[Digit] 2. Pushing dummy input tensor...\n");
    autograd::Tensor dummy_input({1, 1, 28, 28}, false);
    X_input = tape->push_tensor(dummy_input, false);

    printf("[Digit] 3. Tracing forward pass (Graph Compilation)...\n");
    predictions = model.forward(X_input);
    printf("[Digit] 4. Initialization complete!\n");
  }

  val predict(val input_js_array) {
    try {
      std::vector<float> input_data = vecFromJSArray<float>(input_js_array);
      std::copy(input_data.begin(), input_data.end(), X_input->data.data());
      tape->replay_forward();

      float max_val = -std::numeric_limits<float>::infinity();
      for (int i = 0; i < num_classes; ++i) {
        if (predictions->data.data()[i] > max_val)
          max_val = predictions->data.data()[i];
      }

      float sum = 0.0f;
      for (int i = 0; i < num_classes; ++i) {
        probabilities[i] = std::exp(predictions->data.data()[i] - max_val);
        sum += probabilities[i];
      }
      for (int i = 0; i < num_classes; ++i) {
        probabilities[i] /= sum;
      }

      return val::array(probabilities);
    } catch (const std::exception& e) {
      printf("[Digit] C++ Exception in predict: %s\n", e.what());
      return val::null();
    } catch (...) {
      printf("[Digit] Unknown C++ Exception in predict\n");
      return val::null();
    }
  }
};

// ==========================================
// 3. Emscripten Bindings
// ==========================================
EMSCRIPTEN_BINDINGS(nnengine_wasm) {
  class_<WasmClassifier>("WasmClassifier")
      .constructor<int>()
      .function("predict", &WasmClassifier::predict);

  class_<WasmDigitClassifier>("WasmDigitClassifier")
      .constructor<>()
      .function("predict", &WasmDigitClassifier::predict);
}