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
    float frac_part = value - (float)ivalue;

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

Fixed_point Fixed_point::add(const Fixed_point& f) {
    /*
     * 定点加法，返回新定点数，其值为this + f
     */
    // 转换为补码
    unsigned int this_ivalue = ivalue;
    unsigned int this_fvalue = fvalue;
    if(this->sign) {
        this_fvalue = ~(fvalue | 0xffff0000) + 1;   // 先或0xffff0000使高16位为1，然后取反则全为0。再加1
        int c = (int)((this_fvalue >> 16) & 0x1);   // 记录取反加一后的进位
        this_fvalue = this_fvalue & 0x0000ffff;     // 只保留低16位
        this_ivalue = ~(this_ivalue) + c;           // 整数部分和小数部分属于同一个数，整数部分只是这个数的高位。
                                                    // 求补码时，对整个数取反加一，所以只有小数部分需要加一，整数部分
                                                    // 不需要加一，而是要加上小数部分的进位
    }
    unsigned int f_ivalue = f.ivalue;
    unsigned int f_fvalue = f.fvalue;
    if(f.sign) {
        f_fvalue = ~(f.fvalue | 0xffff0000) + 1;
        int c = (int)((f_fvalue >> 16) & 0x1);
        f_fvalue = f_fvalue & 0x0000ffff;
        f_ivalue = ~(f_ivalue) + c;
    }

    // 补码计算
    this_fvalue += f_fvalue;
    int c = (int)((this_fvalue >> 16) & 0x1);
    this_fvalue = this_fvalue & 0x0000ffff;
    this_ivalue += f_ivalue + c;

    // 转换回原码
    int this_sign = 0;
    if(this_ivalue & 0x80000000) {
        this_fvalue = this_fvalue - 1;
        c = (int)((this_fvalue >> 16) & 0x1);
        this_fvalue = (~this_fvalue) & 0x0000ffff;
        this_ivalue = ~(this_ivalue - c);
        this_sign = 1;
    }

    // 存入结果
    Fixed_point result{0};
    result.sign = this_sign;
    result.ivalue = this_ivalue;
    result.fvalue = this_fvalue;
    return result;
}

Fixed_point Fixed_point::mult(const Fixed_point& f) {
    /*
     * 定点乘法。返回新定点数，其值为this * f
     *
     * 每个数都有32位整数，16位小数共48位
     * 算法：
     * temp0[95:0] = {32'h0, this[47:0] * f[15:0]}
     * temp1[95:0] = {16'h0, this[47:0] * f[31:16], 16'h0}
     * temp2[95:0] = {this[47:0] * f[47:32], 32'h0}
     * res = temp0 + temp1 + temp2
     */
    // 将this的值读入ull，低48bit有效
    unsigned long long llthis = ivalue;
    llthis = llthis << 16;
    llthis = llthis | (fvalue & 0x0000ffff);

    // 将f的低中高16bit分别读入ull
    unsigned long long f0 = f.fvalue & 0x0000ffff;
    unsigned long long f1 = f.ivalue & 0x0000ffff;
    unsigned long long f2 = (f.ivalue >> 16) & 0x0000ffff;

    // 将this与3段f依次相乘
    unsigned long long t0 = llthis * f0;
    unsigned long long t1 = llthis * f1;
    unsigned long long t2 = llthis * f2;

    // 将结果读入连续内存
    unsigned int temp0[3] = {0};        // 每个uint存32bit，存96bit共需3个int
    unsigned int temp1[3] = {0};
    unsigned int temp2[3] = {0};
    // temp0[95:0] = {32'h0, t0}
    temp0[0] = t0 & 0xffffffff;
    temp0[1] = (t0 >> 32) & 0xffffffff;
    // temp1[95:0] = {16'h0, t1, 16'h0};
    temp1[0] = ((t1 & 0x0000ffff) << 16) & 0xffff0000;
    temp1[1] = (t1 >> 16) & 0xffffffff;
    temp1[2] = (t1 >> 48) & 0x0000ffff;
    // temp2[95:0] = {t2, 32'h0};
    temp2[1] = t2 & 0xffffffff;
    temp2[2] = (t2 >> 32) & 0xffffffff;

    // 将3个字节串相加
    unsigned int res[3] = {0};
    unsigned long long s = 0;
    int c = 0;
    for(int i = 0; i<3; i++) {
        s = (unsigned long long)temp0[i] + temp1[i] + temp2[i] + c;
        c = (int)(s >> 32);
        s = s & 0xffffffff;
        res[i] = s;
    }

    // 此时得到的结果保存在res中，其中64位整数，32位小数。只保留低32位整数和高16位小数
    Fixed_point result{0};
    result.sign = this->sign ^ f.sign;
    result.ivalue = res[1];
    result.fvalue = (res[0] >> 16) & 0x0000ffff;
    return result;
}

int Fixed_point::to_int() {
    /*
     * 返回定点数取整的值
     */
    return (sign == 1) ? -((int)ivalue) : (int)ivalue;
}

void Fixed_point::print() {
    /*
     * 打印定点数
     */
    printf("binary value: ");
    if(sign) {
        printf("-");
    }
    for(int i = 31; i>=0; i--) {
        printf("%x", ((ivalue) >> i) & 0x1);
    }
    printf(".");
    for(int i = 15; i>=0; i--) {
        printf("%x", ((fvalue) >> i) & 0x1);
    }
    printf("\n");

    printf("decimal value: ");
    if(sign) {
        printf("-");
    }
    printf("%u.", ivalue);
    float temp = 0;
    for(int i = 15; i>=0; i--) {
        if(((fvalue >> i) & 0x1) == 1) {
            temp += (float)pow(2, i-16);
        }
    }
    while(temp > 0) {
        temp *= 10;
        printf("%d", (int)temp);
        temp -= (int)temp;
    }
    printf("\n");
}

Fixed_point Fixed_point::operator+(const Fixed_point &f) {
    /*
     * overload +
     */
    return this->add(f);
}

Fixed_point::Fixed_point(double value) {
    /*
     * 使用double初始化
     */
    this->assign(value);
}

void Fixed_point::assign(double value) {
    /*
     * 赋值，使用双精度浮点数赋值
     */
    sign = (value < 0) ? 1 : 0;
    value = fabs(value);
    ivalue = (int)value;
    double frac_part = value - (double)ivalue;

    double frac_temp = 1.0;
    double frac_value = 0;
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

Fixed_point Fixed_point::operator*(const Fixed_point &f) {
    /*
     * overload *
     */
    return this->mult(f);
}

Fixed_point &Fixed_point::operator=(int value) {
    /*
     * overload =
     */
    this->assign(value);
    return *this;
}

Fixed_point &Fixed_point::operator=(float value) {
    /*
     * overload =
     */
    this->assign(value);
    return *this;
}

Fixed_point &Fixed_point::operator=(double value) {
    /*
     * overload =
     */
    this->assign(value);
    return *this;
}

Fixed_point &Fixed_point::operator+=(const Fixed_point &f) {
    /*
     * overload +=
     */
    Fixed_point res = this->add(f);
    *this = res;
    return *this;
}

Fixed_point &Fixed_point::operator*=(const Fixed_point &f) {
    /*
     * overload *=
     */
    Fixed_point res = this->mult(f);
    *this = res;
    return *this;
}

Fixed_point::Fixed_point() {
    /*
     * 零初始化
     */
    sign = 0;
    ivalue = 0;
    fvalue = 0;
}

float Fixed_point::get_value() {
    /*
     * 获取值
     */
    float fpart = 0;
    float temp = 1.0;
    for(int i = 15; i>=0; i--) {
        temp /= 2;
        if((fvalue >> i) & 0x00000001) {
            fpart += temp;
        }
    }
    return sign ? (-((float)ivalue + fpart)) : ((float)ivalue + fpart);
}

Fixed_point::~Fixed_point() = default;