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

    Vdarray<float32> *dst = new Vdarray<float32>{src->size};
    *dst = (src->astype_float32()) / 255.0f;    // 不用担心临时对象在返回后被销毁。虽然它确实会被销毁，但由于Vdarray引用计数，
                                                // 已经将dst的data指向临时对象的data，那么即便临时对象被销毁，其data也不会释放
    return dst;
}