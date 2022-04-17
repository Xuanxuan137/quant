//
// Created by noname on 2021/11/15.
//

#ifndef QUANT_FIXED_POINT_H
#define QUANT_FIXED_POINT_H

#include <cstdio>
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

    Fixed_point();                                      // 零初始化
    explicit Fixed_point(int value);                    // 使用整数初始化
    explicit Fixed_point(float value);                  // 使用浮点数初始化
    explicit Fixed_point(double value);                 // 使用双精度浮点数初始化
    void assign(int value);                             // 使用整数赋值
    void assign(float value);                           // 使用浮点数赋值
    void assign(double value);                          // 使用双精度浮点数赋值
    Fixed_point add(const Fixed_point& f);              // 定点数相加
    Fixed_point mult(const Fixed_point& f);             // 定点数相乘
    int to_int();                                       // 转换为整数
    float get_value();                                  // 获取值
    void print();                                       // 打印值
    ~Fixed_point();                                     // 析构函数
    Fixed_point& operator=(int value);                  // overload =
    Fixed_point& operator=(float value);                // overload =
    Fixed_point& operator=(double value);               // overload =
    Fixed_point operator+(const Fixed_point& f);        // overload +
    Fixed_point operator*(const Fixed_point& f);        // overload *
    Fixed_point& operator+=(const Fixed_point& f);      // overload +=
    Fixed_point& operator*=(const Fixed_point& f);      // overload *=
};


#endif //QUANT_FIXED_POINT_H
