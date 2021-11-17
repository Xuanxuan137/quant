//
// Created by noname on 2021/11/16.
//

#include "quant_tools.h"




void calc_m0_n_input_weight(Fixed_point &coe, int &rshift, float s_x, float s_w, float s_y)
{
    /*
     * 为使用input和weight计算的算子计算m0和n。如conv2d, dense等
     */
    float m = s_x * s_w / s_y;
    int n = -1;
    float fm0;
    do {
        n++;
        fm0 = m * (float)pow(2, n);
    } while(fm0 < 0.5);
    coe = fm0;
    rshift = n;
}


void calc_m0_n_input_input(Fixed_point &coe1, Fixed_point &coe2, int &rshift1, int &rshift2,
                           float s_x1, float s_x2, float s_y)
{
    /*
     * 为使用input1和input2计算的算子计算m0和n。如add, concat等
     */
    float m1 = s_x1 / s_y;
    float fm01;
    if(m1 >= 1) {
        rshift1 = 1;
        do {
            rshift1--;
            fm01 = m1 * (float)pow(2, rshift1);
        } while(fm01 >= 1);
    }
    else {
        rshift1= -1;
        do {
            rshift1++;
            fm01 = m1 * (float)pow(2, rshift1);
        } while(fm01 < 0.5);
    }
    coe1 = fm01;

    float m2 = s_x2 / s_y;
    float fm02;
    if(m2 >= 1) {
        rshift2 = 1;
        do {
            rshift2--;
            fm02 = m2 * (float)pow(2, rshift2);
        } while(fm02 >= 1);
    }
    else {
        rshift2 = -1;
        do {
            rshift2++;
            fm02 = m2 * (float)pow(2, rshift2);
        } while(fm02 < 0.5);
    }
    coe2 = fm02;
}