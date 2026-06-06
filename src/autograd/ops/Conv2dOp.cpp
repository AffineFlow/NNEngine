#include "autograd/ops/Conv2dOp.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mlengine::autograd::ops {

namespace {  // Anonymous namespace isolates im2col helpers to this translation
             // unit
void im2col(const float* data_im, int channels, int height, int width,
            int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h,
            int stride_w, float* data_col) {
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w, h_offset = (c / kernel_w) % kernel_h,
        c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset,
            w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
              data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0.0f;
      }
    }
  }
}

void col2im(const float* data_col, int channels, int height, int width,
            int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h,
            int stride_w, float* data_im) {
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w, h_offset = (c / kernel_w) % kernel_h,
        c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset,
            w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_im[(c_im * height + h_pad) * width + w_pad] +=
              data_col[(c * height_col + h) * width_col + w];
      }
    }
  }
}
}  // namespace

Conv2dOp::Conv2dOp(Tensor* x, Tensor* w, Tensor* bias, Tensor* out,
                   int in_channels, int out_channels, int in_h, int in_w,
                   int kernel_size, int stride, int pad)
    : x_(x),
      w_(w),
      bias_(bias),
      out_(out),
      in_channels_(in_channels),
      out_channels_(out_channels),
      in_h_(in_h),
      in_w_(in_w),
      kernel_size_(kernel_size),
      stride_(stride),
      pad_(pad) {}

void Conv2dOp::forward() {
  int batch = x_->data.rows();
  int out_h = (in_h_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int out_w = (in_w_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int col_rows = in_channels_ * kernel_size_ * kernel_size_;
  int col_cols = out_h * out_w;

  if (out_->data.rows() != batch ||
      out_->data.cols() != out_channels_ * col_cols) {
    out_->data.resize(batch, out_channels_ * col_cols);
    if (out_->requires_grad) {
      out_->grad.resize(batch, out_channels_ * col_cols);
      out_->grad.setZero();
    }
  }
  cols_.resize(batch);

#pragma omp parallel for
  for (int b = 0; b < batch; ++b) {
    cols_[b].resize(col_rows, col_cols);
    im2col(x_->data.row(b).data(), in_channels_, in_h_, in_w_, kernel_size_,
           kernel_size_, pad_, pad_, stride_, stride_, cols_[b].data());
    Eigen::Map<mlengine::MatrixRM> out_map(out_->data.row(b).data(),
                                           out_channels_, col_cols);
    out_map.noalias() = w_->data * cols_[b];
    out_map.colwise() += bias_->data.row(0).transpose();
  }
}

void Conv2dOp::backward() {
  int batch = x_->data.rows();
  int out_h = (in_h_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int out_w = (in_w_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int col_cols = out_h * out_w;

  mlengine::MatrixRM dW =
      mlengine::MatrixRM::Zero(w_->data.rows(), w_->data.cols());
  mlengine::MatrixRM db = mlengine::MatrixRM::Zero(1, bias_->data.cols());

#pragma omp parallel
  {
    mlengine::MatrixRM local_dW =
        mlengine::MatrixRM::Zero(w_->data.rows(), w_->data.cols());
    mlengine::MatrixRM local_db =
        mlengine::MatrixRM::Zero(1, bias_->data.cols());

#pragma omp for
    for (int b = 0; b < batch; ++b) {
      Eigen::Map<mlengine::MatrixRM> dout_map(out_->grad.row(b).data(),
                                              out_channels_, col_cols);
      local_db += dout_map.rowwise().sum().transpose();
      local_dW.noalias() += dout_map * cols_[b].transpose();

      if (x_->requires_grad) {
        mlengine::MatrixRM dcol = w_->data.transpose() * dout_map;
        x_->grad.row(b).setZero();
        col2im(dcol.data(), in_channels_, in_h_, in_w_, kernel_size_,
               kernel_size_, pad_, pad_, stride_, stride_,
               x_->grad.row(b).data());
      }
    }
#pragma omp critical
    {
      if (w_->requires_grad) w_->grad += local_dW;
      if (bias_->requires_grad) bias_->grad += local_db;
    }
  }
}
}  // namespace mlengine::autograd::ops