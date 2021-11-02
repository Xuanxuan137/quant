//
// Created by noname on 2021/11/2.
//

#include "preprocess.h"


Vdarray<float32>* preprocess(Vdarray<uint8>* src)
{
    /*
     * 图像预处理函数
     * 此函数的预处理方法： img /= 255.0
     */
    if(src == nullptr) {
        return nullptr;
    }

    std::cout << "Hello world\n";
    return nullptr;
}