//
// Created by noname on 2021/11/2.
//

#include "preprocess.h"


Tensor<float32>* preprocess(Tensor<uint8>* src)
{
    /*
     * 图像预处理函数
     * 此函数的预处理方法： img /= 255.0
     */
    if(src == nullptr) {
        return nullptr;
    }

    Tensor<float32> *dst = new Tensor<float32>{src->size};
    *dst = (src->astype_float32()) / 255.0f;    // 不用担心临时对象在返回后被销毁。虽然它确实会被销毁，但由于Tensor引用计数，
                                                // 已经将dst的data指向临时对象的data，那么即便临时对象被销毁，其data也不会释放
    return dst;
}