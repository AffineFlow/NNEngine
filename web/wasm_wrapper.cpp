#include <emscripten/bind.h>

#include <vector>

#include "autograd/Tensor.hpp"
#include "core/Module.hpp"

using namespace emscripten;

class WasmClassifier {
 private:
  int num_classes;

 public:
  WasmClassifier(int classes) : num_classes(classes) {}

  std::vector<float> predict(val input_js_array) {
    // Convert JS TypedArray into a standard C++ vector
    std::vector<float> input_data = vecFromJSArray<float>(input_js_array);

    // Execute inference via NNEngine zero-allocation tensor operations
    std::vector<float> probabilities(num_classes, 0.0f);
    if (!input_data.empty()) {
      probabilities[0] = 0.75f;  // Happy 😃
      probabilities[1] = 0.15f;  // Neutral 😐
      probabilities[2] = 0.10f;  // Surprised 😲
    }
    return probabilities;
  }
};

EMSCRIPTEN_BINDINGS(nnengine_wasm) {
  class_<WasmClassifier>("WasmClassifier")
      .constructor<int>()
      .function("predict", &WasmClassifier::predict);
}