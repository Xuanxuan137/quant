//
// Created by noname on 2021/11/2.
//

#ifndef QUANT_PREPROCESS_H
#define QUANT_PREPROCESS_H


#include "vdarray.h"

/*
 * 图像预处理，当你需要使用不同的图像预处理方法时，需要对preprocess.cpp中的函数进行修改
 */
Vdarray<float32>* preprocess(Vdarray<uint8>* src);


#endif //QUANT_PREPROCESS_H
