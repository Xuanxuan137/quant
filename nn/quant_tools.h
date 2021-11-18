//
// Created by noname on 2021/11/16.
//

#ifndef QUANT_QUANT_TOOLS_H
#define QUANT_QUANT_TOOLS_H

#include "fixed_point.h"
#include "tensor.h"


int clip(int x, int min, int max);
void calc_m0_n_input_weight(Fixed_point &coe, int &rshift, float s_x, float s_w, float s_y);
void calc_m0_n_input_input(Fixed_point &coe1, Fixed_point &coe2, int &rshift1, int &rshift2,
                           float s_x1, float s_x2, float s_y);
void quant(Tensor<uint8> &dst, Tensor<float32> &src, float scale, int zero, int qmin, int qmax);

#endif //QUANT_QUANT_TOOLS_H
