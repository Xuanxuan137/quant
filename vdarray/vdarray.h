//
// Created by noname on 2021/10/23.
//

#ifndef QUANT_VDARRAY_H
#define QUANT_VDARRAY_H

/*
 * 设计思路：
 * Vdarray(Variable dimension array)是一个用来存储变维数据的类
 * 我们不在Vdarray对象中存储数据, 而是存储一个名为data的指针
 * 此外，还在对象中以vector形式存储尺寸
 *
 * 构造Vdarray:
 * 1. 不给定尺寸：仅创建对象，数据均设为0或null
 * 2. 给定尺寸：申请空间data，同时记录空间首地址mem_addr。size设为输入的size。对mem_addr引用计数
 * 3. 拷贝构造：函数返回对象时会自动调用拷贝构造。将新对象data指向旧对象data，对mem_addr引用计数。cut如果大于0则减1
 * 
 * 析构Vdarray:
 * mem_addr查找引用计数，释放空间
 *
 * 重载 =
 * 1. 当左值为对象(cut==0)时：将左值data指向右值data。mem_addr引用计数。cut为0，其余复制
 * 2. 当左值为对象截取(cut==1)时：检查左值右值size必须相同。将右值data指向的数据复制到左值data指向的地址(真复制)。
 *    由于没有指针变化，不需要改变引用计数
 * 3. 我们希望截取到is_num时，能直接对其赋值。所以当左值is_num时，右值可以为数值。
 *
 * 重载[]
 * -- 使用[int]取数组的一部分，称之为截取
 * -- gcc 使用-fno-elide-constructors参数时，函数返回对象时会自动调用一次拷贝构造
 *        使用该返回值创建对象(ClassA a = func())时会再调用一次拷贝构造
 * 使用[]截取：创建一个对象，使其data指向截取后的位置，但mem_addr仍指向原位置，并引用计数。将其cut设为2。返回这个对象
 *      当截取到只剩一个数字时(如array(2,3,4)，使用array[0][0][0]截取)，我们希望能直接返回一个数字，但由于C++的限制，
 *      无法做到像python那样灵活，所以我们仍只能返回一个对象，但此时我们会将这个对象标记为is_num，并可使用to_num()提取这个数
 *      -- mem_addr指向原位置的理由：如果a是b的截取，a的数据首地址与b不同，a的mem_addr不指向b的首地址
 *         那么当b释放时，由于a的引用计数没有与b计在同一地址，当b的引用计数减少时，无法发现a仍在使用b申请的内存
 *         会直接将b申请的内存释放掉。但此时a仍在使用。将a的计数地址与b计同一地址可解决此问题
 *      -- cut设为2的理由：在使用=时，我们希望对于对象左值和对象截取左值采用不同的操作，所以需要对对象和对象截取进行区分
 *         设计为：对象的cut为0，对象截取的cut为1。即[]返回的对象的cut为1，其余为0
 *         在[]中创建临时对象时，将其cut设为2。返回时自动调用拷贝构造，检测到2，发现是[]返回的对象，将cut减1。
 *         这样返回到外层函数的对象截取的cut就为1。
 *         当使用此对象截取创建对象时，仍会调用拷贝构造，再将cut减1，这样新对象就是普通对象
 *         当使用此对象作为右值时，会调用operator=
 * 
 * 
 * 引用计数
 * 由data管理数据，mem_addr管理内存
 * data指向本对象数据的起点
 * 由于使用[]截取后，data可能会改变，不再指向申请的内存的起点，但截取前后使用的是同一次申请的内存，该内存也应该且只能释放一次
 * 所以截取后，让mem_addr仍指向申请的内存的起点。当没有mem_addr指向该地址时，才释放这块内存
 */


#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>
#include <type_traits>

typedef char int8;
typedef unsigned char uint8;
typedef int int32;
typedef unsigned int uint32;
typedef float float32;
typedef double float64;


template <typename T>
class Vdarray {
public:
    // 属性
    T * data;                   // 数据空间
    std::vector<int> size;      // 数据尺寸
    int cut;                    // 是否是截取

    // 构造与析构
    Vdarray();                                              // constructor
    explicit Vdarray(const std::vector<int>& size);         // constructor with input shape
    Vdarray(const Vdarray<T> &src);                         // 拷贝构造函数
    ~Vdarray();                                             // 析构函数

    // 数组处理
    T to_num();                                             // 返回数值
    void set_zero();                                        // data置0
    void set_rand();                                        // data置随机值
    Vdarray<T> reshape(const std::vector<int>& new_size);   // reshape
    Vdarray<T> transpose(const std::vector<int>& new_order);// transpose
    void print();                                           // print
    std::vector<int> shape();                               // shape
    int len();                                              // 返回数据空间元素数量
    int argmax();                                           // argmax
    Vdarray<int> argmax(int axis);                          // argmax
    Vdarray<T> broadcast_to(const std::vector<int> &size);  // broadcast_to

    // 数学运算
    Vdarray<T> divide(float64 divisor);                     // 除法
    Vdarray<T> true_divide(float64 divisor);                // 除法
    Vdarray<T> floor_divide(float64 divisor);               // 向下取整除法
    Vdarray<T> divide(Vdarray<T> divisor);                  // 除法
    Vdarray<T> true_divide(Vdarray<T> divisor);             // 除法
    Vdarray<T> floor_divide(Vdarray<T> divisor);            // 向下取整乘法

    // 运算符重载
    Vdarray<T> operator[](int index);                       // overload []
    Vdarray<T>& operator=(Vdarray<T> array);                // overload =
    Vdarray<T>& operator=(T value);                         // overload =
    Vdarray<T> operator/(float64 divisor);                  // overload /
    Vdarray<T> operator/(Vdarray<T> divisor);               // overload /

private:
    T * mem_addr;               // 记录内存地址
    bool is_num;                // 是否是数值

    static std::map<T*, int> counter;   // reference counter
};
template <typename T>
std::map<T*, int> Vdarray<T>::counter;   // reference counter


void array_add_1(int array[], const std::vector<int> &size);    // 数组自增
void print_size(const std::vector<int> &size);                  // 打印Vdarray的size


namespace VDarray {
    Vdarray<float> zeros(const std::vector<int>& size);
    Vdarray<float> rand(const std::vector<int>& size);
}

#include "vdarray_impl.h"

#endif //QUANT_VDARRAY_H
