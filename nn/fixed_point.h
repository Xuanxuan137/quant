//
// Created by noname on 2021/11/15.
//

#ifndef QUANT_FIXED_POINT_H
#define QUANT_FIXED_POINT_H

#include <cstdlib>
#include <cmath>

/*
 * 定点数
 * 共48位，其中32位整数部分，16位小数部分。
 * 还有1位符号位
 */

class Fixed_point {
public:
    int sign;                   // 符号位      1为负，0为正
    unsigned int ivalue;        // 整数部分
    unsigned int fvalue;        // 小数部分，只使用低16位

    Fixed_point(int value);                     // 使用整数初始化
    Fixed_point(float value);                   // 使用浮点数初始化
    void assign(int value);                     // 使用整数赋值
    void assign(float value);                   // 使用浮点数赋值
    Fixed_point add(Fixed_point f);             // 定点数相加
    Fixed_point mult(Fixed_point f);            // 定点数相乘
    int to_int();                               // 转换为整数
    void print();                               // 打印值
    ~Fixed_point();                             // 析构函数
};


#endif //QUANT_FIXED_POINT_H
