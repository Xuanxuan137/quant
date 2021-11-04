//
// Created by noname on 2021/11/3.
//

#ifndef QUANT_FUNCTIONAL_H
#define QUANT_FUNCTIONAL_H


#include "tensor.h"

namespace functional {
    Tensor<float32> conv2d(Tensor<float32> *input, Tensor<float32> *weight, Tensor<float32> *bias= nullptr,
                           std::vector<int> stride=std::vector<int>{1,1},
                           std::vector<int> padding=std::vector<int>{0,0},
                           std::vector<int> dilation=std::vector<int>{1,1});
}


#endif //QUANT_FUNCTIONAL_H
