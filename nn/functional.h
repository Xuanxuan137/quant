//
// Created by noname on 2021/11/3.
//

#ifndef QUANT_FUNCTIONAL_H
#define QUANT_FUNCTIONAL_H


#include <thread>
#include <cassert>
#include <cblas.h>
#include "tensor.h"
#include "util.h"
#include "fixed_point.h"
#include "quant_tools.h"


namespace functional {
    // float32算子
    Tensor<float32> conv2d(Tensor<float32> *input, Tensor<float32> *weight, Tensor<float32> *bias= nullptr,
                           const std::vector<int>& stride=std::vector<int>{1,1},
                           const std::vector<int>& padding=std::vector<int>{0,0},
                           const std::vector<int>& dilation=std::vector<int>{1,1});
    Tensor<float32> relu(Tensor<float32> *input);
    Tensor<float32> padding(Tensor<float32> *input, const std::vector<int>& padding_size);
    Tensor<float32> maxpool2d(Tensor<float32> *input, const std::vector<int>& kernel_size,
                              std::vector<int> stride=std::vector<int>{-1,-1},
                              const std::vector<int>& padding_size=std::vector<int>{0,0},
                              const std::vector<int>& dilation=std::vector<int>{1,1});
    Tensor<float32> flatten(Tensor<float32> *input);
    Tensor<float32> dense(Tensor<float32> *input, Tensor<float32> *weight, Tensor<float32> *bias= nullptr);
    Tensor<float32> add(Tensor<float32> *input1, Tensor<float32> *input2);
    Tensor<float32> concat(Tensor<float32> *input1, Tensor<float32> *input2, int dim=0);
    Tensor<float32> batch_norm2d(Tensor<float32> *input, Tensor<float32> *running_mean,
                               Tensor<float32> *running_var, Tensor<float32> *weight,
                               Tensor<float32> *bias, float eps);
    Tensor<float32> avgpool2d(Tensor<float32> *input, const std::vector<int>& kernel_size,
                              std::vector<int> stride=std::vector<int>{-1,-1},
                              const std::vector<int>& padding_size=std::vector<int>{0,0});
    Tensor<float32> dropout(Tensor<float32> *input, const float p);
    void im2col(float32 * data_col, float32 * data_im, int height, int width, int channels_col, 
                int height_col, int width_col, int kernel_h, int kernel_w, int stride_h, int stride_w, 
                int pad_h, int pad_w, int dilation_h, int dilation_w);

    // uint8算子
    Tensor<uint8> qconv2d(Tensor<uint8> *input,
                          int zero_x, int zero_w, int zero_b, int zero_y,
                          Fixed_point coe, int rshift, int qmin, int qmax,
                          Tensor<uint8> *weight, Tensor<int32> *bias= nullptr,
                          const std::vector<int>& stride=std::vector<int>{1,1},
                          const std::vector<int>& padding=std::vector<int>{0,0},
                          const std::vector<int>& dilation=std::vector<int>{1,1});
    Tensor<uint8> qrelu(Tensor<uint8> *input, int zero, int qmax);
    Tensor<uint8> qpadding(Tensor<uint8> *input, const std::vector<int>& padding_size, int zero);
    Tensor<uint8> qmaxpool2d(Tensor<uint8> *input, int zero, const std::vector<int>& kernel_size,
                             std::vector<int> stride=std::vector<int>{-1,-1},
                             const std::vector<int>& padding_size=std::vector<int>{0,0},
                             const std::vector<int>& dilation=std::vector<int>{1,1});
    Tensor<uint8> qflatten(Tensor<uint8> *input);
    Tensor<uint8> qdense(Tensor<uint8> *input, int zero_x, int zero_w, int zero_b, int zero_y,
                         Fixed_point coe, int rshift, int qmin, int qmax,
                         Tensor<uint8> *weight, Tensor<int32> *bias= nullptr);
    Tensor<uint8> qadd(Tensor<uint8> *input1, Tensor<uint8> *input2, int zero_x1, int zero_x2,
                       int zero_y, Fixed_point coe1, Fixed_point coe2, int rshift1, int rshift2,
                       int qmin, int qmax);
    Tensor<uint8> qconcat(Tensor<uint8> *input1, Tensor<uint8> *input2, int zero_x1, int zero_x2,
                          int zero_y, Fixed_point coe1, Fixed_point coe2, int rshift1, int rshift2,
                          int qmin, int qmax, int dim=0);
    Tensor<uint8> qavgpool2d(Tensor<uint8> *input, int zero, const std::vector<int>& kernel_size,
                             std::vector<int> stride=std::vector<int>{-1,-1},
                             const std::vector<int>& padding_size=std::vector<int>{0,0});
    void im2col(uint8 * data_col, uint8 * data_im, int height, int width, int channels_col, 
                int height_col, int width_col, int kernel_h, int kernel_w, int stride_h, int stride_w, 
                int pad_h, int pad_w, int dilation_h, int dilation_w, int zero);
}


#endif //QUANT_FUNCTIONAL_H
