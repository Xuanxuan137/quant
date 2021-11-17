//
// Created by noname on 2021/11/3.
//

#ifndef QUANT_FUNCTIONAL_H
#define QUANT_FUNCTIONAL_H

#include <thread>
#include <cassert>
#include "tensor.h"
#include "util.h"


namespace functional {
    Tensor<float32> conv2d(Tensor<float32> *input, Tensor<float32> *weight, Tensor<float32> *bias= nullptr,
                           const std::vector<int>& stride=std::vector<int>{1,1},
                           const std::vector<int>& padding=std::vector<int>{0,0},
                           const std::vector<int>& dilation=std::vector<int>{1,1});
    Tensor<float32> relu(Tensor<float32> *input);
    Tensor<float32> padding(Tensor<float32> *input, const std::vector<int>& padding_size);
    Tensor<float32> maxpool2d(Tensor<float32> *input, const std::vector<int>& kernel_size,
                              std::vector<int> stride=std::vector<int>{-1,-1},
                              const std::vector<int>& padding=std::vector<int>{0,0},
                              const std::vector<int>& dilation=std::vector<int>{1,1});
    Tensor<float32> flatten(Tensor<float32> *input);
    Tensor<float32> dense(Tensor<float32> *input, Tensor<float32> *weight, Tensor<float32> *bias= nullptr);
    Tensor<float32> add(Tensor<float32> *input1, Tensor<float32> *input2);
    Tensor<float32> concat(Tensor<float32> *input1, Tensor<float32> *input2, int dim=0);
    Tensor<float32> batch_norm2d(Tensor<float32> *input, Tensor<float32> *running_mean,
                               Tensor<float32> *running_var, Tensor<float32> *weight,
                               Tensor<float32> *bias, float eps, float momentum);
}


#endif //QUANT_FUNCTIONAL_H
