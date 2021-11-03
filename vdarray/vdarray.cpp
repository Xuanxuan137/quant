//
// Created by noname on 2021/10/23.
//

#include "vdarray.h"

void array_add_1(int array[], const std::vector<int> &size)
{
    /*
     * 给出一个数组array，和数组每一位的进制size。让数组自增一
     * 常用于Vdarray按下标遍历。调用此函数可将下标加一
     */
    int dim = (int)size.size();
    int p = dim-1;
    array[p]++;
    while(true) {
        if(array[p] >= size[p]) {
            array[p] = 0;
            p--;
            if(p < 0) {
                break;
            }
            array[p]++;
        }
        else {
            break;
        }
    }
}

void print_size(const std::vector<int> &size)
{
    /*
     * 打印一个Vdarray的size
     */
    for(const int &i: size) {
        printf("%d ", i);
    }
    printf("\n");
}


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