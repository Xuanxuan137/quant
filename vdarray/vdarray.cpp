//
// Created by noname on 2021/10/23.
//

#include "vdarray.h"




Vdarray<float> VDarray::zeros(const std::vector<int>& size)
{
    /*
     * Create a Vdarray object. Let its shape be size, all value be 0
     */
    Vdarray<float> array(size);
    array.set_zero();
    return array;
}

Vdarray<float> VDarray::rand(const std::vector<int>& size)
{
    /*
     * Create a Vdarray object. Let its shape be size, value be random value between -1 to 1.
     */
    Vdarray<float> array(size);
    array.set_rand();
    return array;
}