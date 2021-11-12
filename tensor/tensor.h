//
// Created by noname on 2021/10/23.
//

#ifndef QUANT_TENSOR_H
#define QUANT_TENSOR_H

/*
 * 设计思路：
 * Tensor 是一个用来存储张量的类
 * 我们不在Tensor对象中存储数据, 而是存储一个名为data的指针
 * 此外，还在对象中以vector形式存储尺寸
 *
 *
 * 关于Tensor具体设计的一些说明：
 * 设计宗旨：
 * 提供方便的张量计算库。尽可能提高性能。
 * 进行简单的张量操作时，尽可能减少内存的复制。一方面减少内存占用，另一方面提高速度。
 *
 * 具体设计：
 * 一. 对属性的解释：
 * 1. data: 数据指针，指向张量中存储数据的地方。
 * 2. mem_addr: 内存地址，指向张量内存的起始地址(后面解释mem_addr和data的区别)
 * 3. size: 张量拥有可变的维度和尺寸。使用一个vector存储尺寸
 * 4. cut: 标记当前张量是否是其他张量的截取(即在程序中，是否以t[i]的形式出现)
 * 5. is_num: 标记当前张量是否是数值(对于n维张量，通过n次[下标]的方式取出数值)
 * 6. counter: 引用计数器
 * 引用计数：
 *      由于为了减少内存空间占用和提高性能，所以在对张量进行简单的操作，如对象复制，数据截取，变形等不会改变数据的值的操作时，
 *      我们仅将创建的新张量的data指向旧张量的data，而不为新张量分配新的数据空间。即让新张量和旧张量共享数据空间，但它们可能
 *      拥有不同的尺寸，维度，数据长度等属性。
 *      但这会带来一个问题：由于新旧张量共享数据空间，我们必须保证旧张量析构时，其数据空间不被释放，因为新张量仍在使用它。
 *      考虑到'[]'的存在，即便两个张量共享数据空间，其实际使用的数据空间起始地址(即data)仍可能不同，所以我们必须为每个张量新建一个
 *      mem_addr属性，记录此张量的数据空间的真正起始地址，用于管理内存的释放。与此同时，我们需要引入引用计数方法，根据mem_addr
 *      进行引用计数。对于某一块内存，只有当所有张量都不使用它时(所有现存张量的mem_addr的值都不为它)，才将它释放。
 *      引用计数的计算方式：使用map记录所有出现的mem_addr，和每个mem_addr的出现次数。创建新的张量时，会对mem_addr次数加一(或
 *      加入map中)。析构张量时，会对mem_addr减一(或从map中删除)
 *      注意：共享数据空间会带来一个问题：如果两个张量共享数据空间，那么对其中一个的数据进行修改时，另一个的数据也会改变
 *
 * 二. 构造与析构Tensor:
 * 1. 不给定尺寸：仅创建对象，数据均设为0或null
 * 2. 给定尺寸：申请空间data，同时记录空间首地址mem_addr。size设为输入的size。对mem_addr引用计数
 * 3. 拷贝构造：函数返回对象时会自动调用拷贝构造。共享数据空间
 * 4. 析构：mem_addr查找引用计数，释放空间
 *
 * 三. 数组处理:
 * 1. print(): 打印所有数据
 * 2. len(): 获取张量的元素个数
 * 3. shape(): 获取张量的形状
 * 4. reshape(): 创建新张量，使其形状为变形后的形状。共享数据空间
 * 5. transpose(): 创建新张量，使其形状和数据顺序为transpose后的样子
 * 6. broadcast_to(): 创建新张量，使其形状为广播后的样子
 * 7. deep_copy(): 创建新张量，复制原张量数据空间内的数据
 *
 * 四. 数据处理:
 * 1. to_num(): n维张量使用[]进行n次截取后，变为数值(is_num->true)。此时可使用to_num()取出这个数值
 * 2. set_zero(): 将原张量中数据设为0
 * 3. set_rand(): 将原张量中数据设为随机值
 * 4. argmax(): 计算张量中所有数据中最大值的下标(此下标为相对于data的偏移)
 * 5. argmax(axis): 计算张量中除axis外其他维度在axis轴上的最大值的下标。结果返回一个张量
 * 6. astype_uint8(): 创建新张量，使其值为原张量转为uint8后的值
 * 7. astype_int32(): 创建新张量，使其值为原张量转为int32后的值
 * 8. astype_float32(): 创建新张量，使其值为原张量转为float32后的值
 *
 * 五. 数学运算:
 * 1. add(): 创建新张量。计算加法
 * 2. divide(): 创建新张量。计算除法
 * 3. true_divide(): 创建新张量。计算除法
 * 4. floor_divide(): 创建新张量。计算向下取整除法
 * 5. dot(): 创建新张量。矩阵乘法
 *
 * 六. 运算符重载：
 * 1. =
 *      1.1 当左值为对象(cut==0)时：将左值data指向右值data。共享数据空间
 *      1.2 当左值为对象截取(cut==1)时：将右值data指向的数据复制到左值data指向的地址(真复制)。
 *      1.3 我们希望截取到is_num时，能直接对其赋值。所以当左值is_num时，右值可以为数值。
 * 2. []
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
 * 3. +: 创建新张量。+
 * 4. +=: 在原张量的基础上+
 * 5. /: 创建新张量。+
 */


#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>
#include <type_traits>
#include <thread>
#include <unistd.h>
#include <cassert>
#include <cmath>

typedef char int8;
typedef unsigned char uint8;
typedef int int32;
typedef unsigned int uint32;
typedef float float32;
typedef double float64;


template <typename T>
class Tensor {
public:
    // 属性
    T * data;                   // 数据空间
    std::vector<int> size;      // 数据尺寸
    int cut;                    // 是否是截取

    // 构造与析构
    Tensor();                                               // constructor
    explicit Tensor(const std::vector<int>& size);          // constructor with input shape
    Tensor(const Tensor<T> &src);                           // 拷贝构造函数
    ~Tensor();                                              // 析构函数

    // 数组处理
    void print();                                           // print
    int len();                                              // 返回数据空间元素数量
    std::vector<int> shape();                               // shape
    Tensor<T> reshape(const std::vector<int>& new_size);    // reshape
    Tensor<T> transpose(const std::vector<int>& new_order); // transpose
    Tensor<T> broadcast_to(const std::vector<int> &size);   // broadcast_to
    Tensor<T> deep_copy();                                  // deep copy
    Tensor<T> concat(Tensor<T> array, int dim = 0);         // concat

    // 数据处理
    T to_num();                                             // 返回数值
    void set_zero();                                        // data置0
    void set_rand();                                        // data置随机值
    int argmax();                                           // argmax
    Tensor<int> argmax(int axis);                           // argmax
    Tensor<uint8> astype_uint8();                           // astype("uint8");
    Tensor<int32> astype_int32();                           // astype("int32")
    Tensor<float32> astype_float32();                       // astype("float32")

    // 数学运算
    Tensor<T> add(T adder);                                 // 加法
    Tensor<T> add(Tensor<T> adder);                         // 加法
    Tensor<T> sub(T subtractend);                           // 减法
    Tensor<T> sub(Tensor<T> subtractend);                   // 减法
    Tensor<T> mult(T multiplier);                           // 乘法
    Tensor<T> mult(Tensor<T> multiplier);                   // 乘法
    Tensor<T> divide(T divisor);                            // 除法
    Tensor<T> true_divide(T divisor);                       // 除法
    Tensor<T> floor_divide(T divisor);                      // 向下取整除法
    Tensor<T> divide(Tensor<T> divisor);                    // 除法
    Tensor<T> true_divide(Tensor<T> divisor);               // 除法
    Tensor<T> floor_divide(Tensor<T> divisor);              // 向下取整乘法
    Tensor<T> elewise_sqrt();                               // 开平方
    Tensor<T> dot(Tensor<T> B);                             // matmul
    T mean();                                               // mean
    Tensor<T> mean(std::vector<int> axis);                  // mean
    T var();                                                // var
    Tensor<T> var(std::vector<int> axis);                   // var

    // 运算符重载
    Tensor<T> operator[](int index);                        // overload []  : Tensor[]
    Tensor<T>& operator=(Tensor<T> array);                  // overload =   : Tensor = Tensor
    Tensor<T>& operator=(T value);                          // overload =   : is_num_Tensor = value
    Tensor<T> operator+(T adder);                           // overload +   : Tensor + value
    Tensor<T> operator+(Tensor<T> adder);                   // overload +   : Tensor + Tensor
    Tensor<T>& operator+=(T adder);                         // overload +=  : Tensor += value
    Tensor<T> operator-(T subtractend);                     // overload -   : Tensor - value
    Tensor<T> operator-(Tensor<T> subtractend);             // overload -   : Tensor - Tensor
    Tensor<T>& operator-=(T subtractend);                   // overload -=  : Tensor -= value
    Tensor<T> operator*(T multiplier);                      // overload *   : Tensor * value
    Tensor<T> operator*(Tensor<T> multiplier);              // overload *   : Tensor * Tensor
    Tensor<T>& operator*=(Tensor<T> multiplier);            // overload *=  : Tensor *= value
    Tensor<T> operator/(T divisor);                         // overload /   : Tensor / value
    Tensor<T> operator/(Tensor<T> divisor);                 // overload /   : Tensor / Tensor
    Tensor<T>& operator/=(Tensor<T> divisor);               // overload /=  : Tensor /= value

private:
    T * mem_addr;               // 记录内存地址
    bool is_num;                // 是否是数值

    static std::map<T*, int> counter;   // reference counter
};
template <typename T>
std::map<T*, int> Tensor<T>::counter;   // reference counter


void array_add_1(int array[], const std::vector<int> &size);    // 数组自增
void print_size(const std::vector<int> &size);                  // 打印Tensor的size
void mt_dot(float * C, float * A, float * B, int mt_M, int mt_K, int mt_N);


#include "tensor_impl.h"

#endif //QUANT_TENSOR_H
