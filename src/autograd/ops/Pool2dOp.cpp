#include "autograd/ops/Pool2dOp.hpp"

#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace affineflow::nn::autograd::ops {

// ==========================================
// MaxPool2dOp Implementation
// ==========================================
MaxPool2dOp::MaxPool2dOp(Tensor* x, Tensor* out, int channels, int in_h,
                         int in_w, int kernel_size, int stride, int pad)
    : x_(x),
      out_(out),
      channels_(channels),
      in_h_(in_h),
      in_w_(in_w),
      kernel_size_(kernel_size),
      stride_(stride),
      pad_(pad) {}

void MaxPool2dOp::forward() {
  int batch = x_->shape[0];
  int out_h = (in_h_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int out_w = (in_w_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int out_area = out_h * out_w;
  int in_area = in_h_ * in_w_;

  if (!out_->has_shape({batch, channels_ * out_area})) {
    out_->resize({batch, channels_ * out_area});
  }

  // Pre-allocate tracking memory once
  if (batch > max_batch_) {
    max_indices_.resize(batch * channels_ * out_area);
    max_batch_ = batch;
  }

  const float* x_data = x_->data.data();
  float* out_data = out_->data.data();
  int* indices_data = max_indices_.data();

#pragma omp parallel for collapse(2)
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels_; ++c) {
      int in_offset = (b * channels_ + c) * in_area;
      int out_offset = (b * channels_ + c) * out_area;

      for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
          float max_val = -std::numeric_limits<float>::infinity();
          int max_idx = -1;

          for (int kh = 0; kh < kernel_size_; ++kh) {
            for (int kw = 0; kw < kernel_size_; ++kw) {
              int ih = oh * stride_ - pad_ + kh;
              int iw = ow * stride_ - pad_ + kw;

              if (ih >= 0 && ih < in_h_ && iw >= 0 && iw < in_w_) {
                int idx = ih * in_w_ + iw;
                float val = x_data[in_offset + idx];
                if (val > max_val) {
                  max_val = val;
                  max_idx = idx;
                }
              }
            }
          }
          out_data[out_offset + oh * out_w + ow] = max_val;
          indices_data[out_offset + oh * out_w + ow] = max_idx;
        }
      }
    }
  }
}

void MaxPool2dOp::backward() {
  if (!x_->requires_grad) return;

  int batch = x_->shape[0];
  int out_h = (in_h_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int out_w = (in_w_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int out_area = out_h * out_w;
  int in_area = in_h_ * in_w_;

  const float* dout_data = out_->grad.data();
  float* dx_data = x_->grad.data();
  const int* indices_data = max_indices_.data();

#pragma omp parallel for collapse(2)
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels_; ++c) {
      int in_offset = (b * channels_ + c) * in_area;
      int out_offset = (b * channels_ + c) * out_area;

      for (int i = 0; i < out_area; ++i) {
        int max_idx = indices_data[out_offset + i];
        if (max_idx != -1) {
          dx_data[in_offset + max_idx] += dout_data[out_offset + i];
        }
      }
    }
  }
}

// ==========================================
// AvgPool2dOp Implementation
// ==========================================
AvgPool2dOp::AvgPool2dOp(Tensor* x, Tensor* out, int channels, int in_h,
                         int in_w, int kernel_size, int stride, int pad)
    : x_(x),
      out_(out),
      channels_(channels),
      in_h_(in_h),
      in_w_(in_w),
      kernel_size_(kernel_size),
      stride_(stride),
      pad_(pad) {}

void AvgPool2dOp::forward() {
  int batch = x_->shape[0];
  int out_h = (in_h_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int out_w = (in_w_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int out_area = out_h * out_w;
  int in_area = in_h_ * in_w_;

  if (!out_->has_shape({batch, channels_ * out_area})) {
    out_->resize({batch, channels_ * out_area});
  }

  const float* x_data = x_->data.data();
  float* out_data = out_->data.data();
  float pool_area = static_cast<float>(kernel_size_ * kernel_size_);

#pragma omp parallel for collapse(2)
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels_; ++c) {
      int in_offset = (b * channels_ + c) * in_area;
      int out_offset = (b * channels_ + c) * out_area;

      for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
          float sum_val = 0.0f;

          for (int kh = 0; kh < kernel_size_; ++kh) {
            for (int kw = 0; kw < kernel_size_; ++kw) {
              int ih = oh * stride_ - pad_ + kh;
              int iw = ow * stride_ - pad_ + kw;

              if (ih >= 0 && ih < in_h_ && iw >= 0 && iw < in_w_) {
                sum_val += x_data[in_offset + ih * in_w_ + iw];
              }
            }
          }
          out_data[out_offset + oh * out_w + ow] = sum_val / pool_area;
        }
      }
    }
  }
}

void AvgPool2dOp::backward() {
  if (!x_->requires_grad) return;

  int batch = x_->shape[0];
  int out_h = (in_h_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int out_w = (in_w_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int out_area = out_h * out_w;
  int in_area = in_h_ * in_w_;

  const float* dout_data = out_->grad.data();
  float* dx_data = x_->grad.data();
  float pool_area = static_cast<float>(kernel_size_ * kernel_size_);

#pragma omp parallel for collapse(2)
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels_; ++c) {
      int in_offset = (b * channels_ + c) * in_area;
      int out_offset = (b * channels_ + c) * out_area;

      for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
          float grad_val = dout_data[out_offset + oh * out_w + ow] / pool_area;

          for (int kh = 0; kh < kernel_size_; ++kh) {
            for (int kw = 0; kw < kernel_size_; ++kw) {
              int ih = oh * stride_ - pad_ + kh;
              int iw = ow * stride_ - pad_ + kw;

              if (ih >= 0 && ih < in_h_ && iw >= 0 && iw < in_w_) {
                dx_data[in_offset + ih * in_w_ + iw] += grad_val;
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace affineflow::nn::autograd::ops