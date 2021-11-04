//
// Created by noname on 2021/10/23.
//

#include "tensor.h"

void array_add_1(int array[], const std::vector<int> &size)
{
    /*
     * 给出一个数组array，和数组每一位的进制size。让数组自增一
     * 常用于Tensor按下标遍历。调用此函数可将下标加一
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
     * 打印一个Tensor的size
     */
    for(const int &i: size) {
        printf("%d ", i);
    }
    printf("\n");
}

