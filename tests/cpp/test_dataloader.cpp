#include <gtest/gtest.h>

#include "autograd/Tensor.hpp"
#include "core/DataLoader.hpp"

using namespace affineengine;
using namespace affineengine::autograd;

TEST(DataLoaderTest, ExactBatching) {
  // 100 samples, batch size 32
  Tensor dataset_x({100, 10});
  Tensor dataset_y({100, 1});

  core::DataLoader loader(dataset_x, dataset_y, 32, false);

  int batch_count = 0;
  int last_batch_size = 0;

  Tensor batch_x({0});
  Tensor batch_y({0});

  while (loader.has_next()) {
    loader.next_batch(batch_x, batch_y);
    batch_count++;
    last_batch_size = batch_x.shape[0];
  }

  EXPECT_EQ(batch_count, 4);  // 32, 32, 32, 4
  EXPECT_EQ(last_batch_size, 4);
}

TEST(DataLoaderTest, ShufflingChangesOrder) {
  Tensor dataset_x({10, 2});  // Small dataset
  dataset_x.data.setConstant(1.0f);

  // Dummy labels 0-9 to track shuffling
  Tensor dataset_y({10, 1});
  for (int i = 0; i < 10; i++) dataset_y.data.data()[i] = (float)i;

  core::DataLoader loader_ordered(dataset_x, dataset_y, 10, false);
  core::DataLoader loader_shuffled(dataset_x, dataset_y, 10, true);

  Tensor batch_x_ord({0}), batch_y_ord({0});
  loader_ordered.next_batch(batch_x_ord, batch_y_ord);

  Tensor batch_x_shuf({0}), batch_y_shuf({0});
  loader_shuffled.next_batch(batch_x_shuf, batch_y_shuf);

  bool is_different = false;
  for (int i = 0; i < 10; i++) {
    if (batch_y_ord.data.data()[i] != batch_y_shuf.data.data()[i]) {
      is_different = true;
      break;
    }
  }

  EXPECT_TRUE(is_different);
}