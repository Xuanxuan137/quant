//
// Created by noname on 2021/10/23.
//

#ifndef QUANT_VDARRAY_H
#define QUANT_VDARRAY_H

/*
 * 设计思路：
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

/*
 * Design ideas：
 * Vdarray(Variable dimension array) is a class used to store variable dimension data.
 * We do not store data in Vdarray object, we just store a pointer in Vdarray object which is called 'data'
 * Besides, we store the shape of data as a vector in Vdarray object
 *
 * Construct Vdarray:
 * 1. No input shape：Only create object，set all attributes to 0 or nullptr
 * 2. Given shape：allocate memory for data，and record the memory start address as mem_addr.
 *                 Set size to the input size。Use reference count for mem_addr
 * 3. Copy constructor：Function will call copy constructor automatically when it returns an object.
 *                      Point the data of new object to the data of old object，reference count mem_addr.
 *                      if cut is greater than 0, then reduce it by 1.
 *
 *
 * Destruct Vdarray:
 * Reference count mem_addr and release memory
 *
 *
 * Overload =
 * 1. When lvalue is an object(cut==0): Point data of lvalue to data of rvalue.
 *          Reference count mem_addr. Set cut to 0 and copy other attribute.
 * 2. When lvalue is an interception of an object(cut==1): Check the size of lvalue and
 *          rvalue must be the same。Copy data of rvalue to data of lvalue(deep copy)。
 *          Since there is no pointer change, no need to change reference count.
 * 3. Since we hope we can assign to an is_num object directly, when the lvalue is is_num,
 *          rvalue can be numerical value
 *
 *
 * Overload []
 * -- Use [int] to take part of the array, which is called interception
 * -- When gcc uses the -fno-elide-constructors option, the copy construction will be automatically
 *          called once when the function returns an object
 *          The copy constructor will be called again when use this return to create an object(class A a = func())
 * Intercept with []：Create an object and make its data points to the memory address after interception,
 *          but mem_addr still points to the original memory address and is reference counted.
 *          Set its cut to 2. Return this object.
 *      When only one numerical value is left(such as array(2,3,4), use array[0][0][0] to intercept),
 *          we hope to return a numerical value directly. But due to the limitations of C++,
 *          it cannot be as flexible as python, so we have to return an object, but at this time we will mark
 *          this object as is_num, and use to_num() to extract this numerical value.
 *      -- The reason why mem_addr points to the original memory address：If A is an interception of B, the
 *          data address of A is different from B, and the mem_addr of A is different from B too, then when B is
 *          released, since the reference count of A is different from the reference of B, the reference counter
 *          will not find that A is still using the memory allocated by B, and the memory will be released.
 *          But at this time this block of memory is still used by A. Set the reference count address of A
 *          to the reference count address of B will solve this problem.
 *      -- The reason why cut is set to 2: When using =, we hope to use different operations for object lvalue
 *          and object interception lvalue, so we need to distinguish between object and object interception
 *         Designed as: The cut of an object is 0; The cut of an object interception is 1。
 *          That is, the cut of the object returned by [] is 1, and the rest are 0
 *         When creating a temporary object in [], set its cut to 2. The copy construction is automatically
 *          invoked when returning. When 2 is detected, it is found that it is the object returned by [],
 *          and cut is reduced by 1. In this way, the cut of an object interception returned to the outer
 *          function is 1.
 *          When using this object interception to create an object, the copy construction will be called again,
 *          and the cut will be reduced by 1 again, so that the new object is a normal object
 *          When using this object as an rvalue, operator= will be called
 *
 *
 * Reference count
 * Data is managed by 'data', and memory is managed by 'mem_addr'
 * 'data' point to the start of the memory address of the object
 * Since after [], data may change and no longer point to the allocated memory, but the memory used before and
 * after interception is allocated once, the memory should and can only release once. So after interception,
 * let mem_addr still point to the start address of the allocated memory. When there is no mem_addr point to
 * this address, release this memory.
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
    T * data;                   // data space
    std::vector<int> size;      // data shape
    int cut;                    // whether is interception
    Vdarray();                                              // constructor
    explicit Vdarray(const std::vector<int>& size);         // constructor with input shape
    Vdarray(const Vdarray<T> &src);                         // copy constructor
    ~Vdarray();                                             // destructor
    Vdarray<T> operator[](int index);                       // overload []
    Vdarray<T>& operator=(Vdarray<T> array);                // overload =
    Vdarray<T>& operator=(T value);                         // overload =
    T to_num();                                             // return numerical value
    void set_zero();                                        // set data to 0
    void set_rand();                                        // set data to random value
    Vdarray<T> reshape(const std::vector<int>& new_size);   // reshape
    Vdarray<T> transpose(const std::vector<int>& new_order);// transpose
    void print();                                           // print
    std::vector<int> shape();                               // shape

private:
    T * mem_addr;               // record data address
    bool is_num;                // whether is numerical value

    static std::map<T*, int> counter;   // reference counter
};
template <typename T>
std::map<T*, int> Vdarray<T>::counter;   // reference counter



namespace VDarray {
    Vdarray<float> zeros(const std::vector<int>& size);
    Vdarray<float> rand(const std::vector<int>& size);
}

#include "vdarray_impl.h"

#endif //QUANT_VDARRAY_H
