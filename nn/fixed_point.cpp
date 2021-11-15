//
// Created by noname on 2021/11/15.
//

#include "fixed_point.h"

Fixed_point::Fixed_point(int value) {
    /*
     * 构造函数，使用整数构造
     */
    this->assign(value);
}

Fixed_point::Fixed_point(float value) {
    /*
     * 构造函数，使用浮点数构造
     */
    this->assign(value);
}

void Fixed_point::assign(int value) {
    /*
     * 赋值。使用整数赋值
     */
    sign = (value < 0) ? 1 : 0;
    ivalue = abs(value);
    fvalue = 0;
}

void Fixed_point::assign(float value) {
    /*
     * 赋值，使用浮点数赋值
     */
    sign = (value < 0) ? 1 : 0;
    value = fabsf(value);
    ivalue = (int)value;
    float frac_part = value - ivalue;

    float frac_temp = 1.0;
    float frac_value = 0;
    for(int i = 15; i>=0; i--) {
        frac_temp /= 2;
        if(frac_value + frac_temp > frac_part) {
            fvalue = fvalue << 1;
            fvalue = fvalue & 0xfffffffe;
        }
        else {
            fvalue = fvalue << 1;
            fvalue = fvalue | 0x00000001;
            frac_value += frac_temp;
        }
    }
}

Fixed_point Fixed_point::add(Fixed_point f) {
    /*
     * 定点加法，返回新定点数，其值为this + f
     */
    // 转换为补码
    unsigned int this_ivalue = ivalue;
    unsigned int this_fvalue = fvalue;
    if(this->sign) {
        this_fvalue = ~(fvalue | 0xffff0000) + 1;   // 先或0xffff0000使高16位为1，然后取反则全为0。再加1
        int c = (this_fvalue >> 16) & 0x1;          // 记录取反加一后的进位
        this_fvalue = this_fvalue & 0x0000ffff;     // 只保留低16位
        this_ivalue = ~(this_ivalue) + c;           // 整数部分和小数部分属于同一个数，整数部分只是这个数的高位。
                                                    // 求补码时，对整个数取反加一，所以只有小数部分需要加一，整数部分
                                                    // 不需要加一，而是要加上小数部分的进位
    }
    unsigned int f_ivalue = f.ivalue;
    unsigned int f_fvalue = f.fvalue;
    if(f.sign) {
        f_fvalue = ~(f.fvalue | 0xffff0000) + 1;
        int c = (f_fvalue >> 16) & 0x1;
        f_fvalue = f_fvalue & 0x0000ffff;
        f_ivalue = ~(f_ivalue) + c;
    }

    // 补码计算
    this_fvalue += f_fvalue;
    int c = (this_fvalue >> 16) & 0x1;
    this_fvalue = this_fvalue & 0x0000ffff;
    this_ivalue += f_ivalue + c;

    // 转换回原码
    char this_sign = 0;
    if(this_ivalue & 0x80000000) {
        this_fvalue = this_fvalue - 1;
        int c = (this_fvalue >> 16) & 0x1;
        this_fvalue = (~this_fvalue) & 0x0000ffff;
        this_ivalue = ~(this_ivalue - c);
        this_sign = 1;
    }

    // 存入结果
    Fixed_point result;
    result.sign = this_sign;
    result.ivalue = this_ivalue;
    result.fvalue = this_fvalue;
    return result;
}

Fixed_point Fixed_point::mult(Fixed_point f) {
    return Fixed_point(0);
}

int Fixed_point::to_int() {
    return 0;
}

void Fixed_point::print() {

}

Fixed_point::~Fixed_point() {

}
