//
// Created by noname on 2021/11/2.
//

#include "preprocess.h"
#include "tensor.h"


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

Tensor<uint8>* qpreprocess(Tensor<uint8>* src)
{
    return src;
}






// Tensor<float32>* preprocess(Tensor<uint8>* src)
// {
//     /*
//      * 图像预处理函数
//      * 此函数的预处理方法： imgRGB = (imgRGB - [123.15, 115.90, 103.06]) / (58.395, 57.12, 57.375)
//      * 适用于imageNet
//      */
//     if(src == nullptr) {
//         return nullptr;
//     }

//     Tensor<float32> *dst = new Tensor<float32>{src->size};
//     int img_num = dst->size[0];
//     for(int i = 0; i<img_num; i++) {
//         (*dst)[i][0] = ((*src)[i][0].astype_float32() - 123.15) / 58.395;
//         (*dst)[i][1] = ((*src)[i][1].astype_float32() - 115.90) / 57.12;
//         (*dst)[i][2] = ((*src)[i][2].astype_float32() - 103.06) / 57.375;
//     }
//     return dst;
// }

// Tensor<uint8>* qpreprocess(Tensor<uint8>* src)
// {
//     /*
//      * 图像预处理函数
//      * 此函数的预处理方法： imgRGB = (imgRGB - [123.15, 115.90, 103.06]) / (58.395, 57.12, 57.375)
//      * 适用于imageNet
//      */
//     if(src == nullptr) {
//         return nullptr;
//     }

//     Tensor<float32> *dst = new Tensor<float32>{src->size};
//     int img_num = dst->size[0];
//     for(int i = 0; i<img_num; i++) {
//         (*dst)[i][0] = ((*src)[i][0].astype_float32() - 123.15) / 58.395 / 0.016631 + 104;
//         (*dst)[i][1] = ((*src)[i][1].astype_float32() - 115.90) / 57.12 / 0.016631 + 104;
//         (*dst)[i][2] = ((*src)[i][2].astype_float32() - 103.06) / 57.375 / 0.016631 + 104;
//     }
//     dst->clip(0, 255);
//     Tensor<uint8>* ret = new Tensor<uint8>{dst->size};
//     *ret = dst->astype_uint8();
//     return ret;
// }