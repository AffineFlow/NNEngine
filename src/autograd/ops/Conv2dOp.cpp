#include "autograd/ops/Conv2dOp.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace affineengine::autograd::ops {
namespace {
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
      pad_(pad) {
#ifdef _OPENMP
  num_threads_ = omp_get_max_threads();
#endif
  thread_dW_.resize(num_threads_);
  thread_db_.resize(num_threads_);
  thread_dcol_.resize(num_threads_);
}

void Conv2dOp::forward() {
  int batch = x_->shape[0];
  int out_h = (in_h_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int out_w = (in_w_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int col_rows = in_channels_ * kernel_size_ * kernel_size_;
  int col_cols = out_h * out_w;

  affineengine::Shape out_shape = {batch, out_channels_ * col_cols};
  if (!out_->has_shape(out_shape)) out_->resize(out_shape);

  if (batch > max_batch_) {
    cols_.resize(batch);
    for (int b = 0; b < batch; ++b) {
      cols_[b].resize(col_rows * col_cols);
    }
    max_batch_ = batch;
  }

  Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      W_mat(w_->data.data(), out_channels_, col_rows);
  Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      B_mat(bias_->data.data(), 1, out_channels_);

#pragma omp parallel for
  for (int b = 0; b < batch; ++b) {
    im2col(x_->data.data() + (b * in_channels_ * in_h_ * in_w_), in_channels_,
           in_h_, in_w_, kernel_size_, kernel_size_, pad_, pad_, stride_,
           stride_, cols_[b].data());

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        col_map(cols_[b].data(), col_rows, col_cols);
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        out_map(out_->data.data() + (b * out_channels_ * col_cols),
                out_channels_, col_cols);

    out_map.noalias() = W_mat * col_map;
    out_map.colwise() += B_mat.row(0).transpose();
  }
}

void Conv2dOp::backward() {
  int batch = x_->shape[0];
  int out_h = (in_h_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int out_w = (in_w_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int col_rows = in_channels_ * kernel_size_ * kernel_size_;
  int col_cols = out_h * out_w;

  Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      W_mat(w_->data.data(), out_channels_, col_rows);

  for (int t = 0; t < num_threads_; ++t) {
    if (thread_dW_[t].size() != out_channels_ * col_rows) {
      thread_dW_[t].resize(out_channels_ * col_rows);
    }
    if (thread_db_[t].size() != out_channels_) {
      thread_db_[t].resize(out_channels_);
    }
    if (x_->requires_grad && thread_dcol_[t].size() != col_rows * col_cols) {
      thread_dcol_[t].resize(col_rows * col_cols);
    }
    thread_dW_[t].setZero();
    thread_db_[t].setZero();
  }

#pragma omp parallel
  {
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif

#pragma omp for
    for (int b = 0; b < batch; ++b) {
      Eigen::Map<
          Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          dout_map(out_->grad.data() + (b * out_channels_ * col_cols),
                   out_channels_, col_cols);

      Eigen::Map<
          Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          col_map(cols_[b].data(), col_rows, col_cols);

      Eigen::Map<
          Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          t_db(thread_db_[tid].data(), 1, out_channels_);
      Eigen::Map<
          Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          t_dw(thread_dW_[tid].data(), out_channels_, col_rows);

      t_db += dout_map.rowwise().sum().transpose();
      t_dw.noalias() += dout_map * col_map.transpose();

      if (x_->requires_grad) {
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
            t_dcol(thread_dcol_[tid].data(), col_rows, col_cols);

        t_dcol.noalias() = W_mat.transpose() * dout_map;

        float* dx_batch_ptr =
            x_->grad.data() + (b * in_channels_ * in_h_ * in_w_);
        std::fill(dx_batch_ptr, dx_batch_ptr + (in_channels_ * in_h_ * in_w_),
                  0.0f);
        col2im(t_dcol.data(), in_channels_, in_h_, in_w_, kernel_size_,
               kernel_size_, pad_, pad_, stride_, stride_, dx_batch_ptr);
      }
    }
  }

  if (w_->requires_grad) {
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        dW_target(w_->grad.data(), out_channels_, col_rows);
    for (int t = 0; t < num_threads_; ++t) {
      Eigen::Map<
          Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          t_dw(thread_dW_[t].data(), out_channels_, col_rows);
      dW_target += t_dw;
    }
  }
  if (bias_->requires_grad) {
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        db_target(bias_->grad.data(), 1, out_channels_);
    for (int t = 0; t < num_threads_; ++t) {
      Eigen::Map<
          Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          t_db(thread_db_[t].data(), 1, out_channels_);
      db_target += t_db;
    }
  }
}
}  // namespace affineengine::autograd::ops