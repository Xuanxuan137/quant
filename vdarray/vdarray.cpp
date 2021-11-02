//
// Created by noname on 2021/10/23.
//

#include "vdarray.h"




Vdarray<float> VDarray::zeros(const std::vector<int>& size)
{
    /*
     * 创建Vdarray对象，使其size为size，值为0
     */
    Vdarray<float> array(size);
    array.set_zero();
    return array;
}

Vdarray<float> VDarray::rand(const std::vector<int>& size)
{
    /*
     * 创建Vdarray对象，使其size为size，值为-1～1的随机值
     */
    Vdarray<float> array(size);
    array.set_rand();
    return array;
}