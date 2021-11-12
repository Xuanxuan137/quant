//
// Created by noname on 2021/10/25.
//

#ifndef QUANT_TENSOR_IMPL_H
#define QUANT_TENSOR_IMPL_H

#include "tensor.h"

template<typename T>
Tensor<T>::Tensor() {
    /*
     * Tensor 构造函数：
     * 只创建对象，不分配空间
     */
    if(!std::is_same<T, int>::value &&
       !std::is_same<T, unsigned int>::value &&
       !std::is_same<T, long int>::value &&
       !std::is_same<T, unsigned long int>::value &&
       !std::is_same<T, long long int>::value &&
       !std::is_same<T, unsigned long long int>::value &&
       !std::is_same<T, float>::value &&
       !std::is_same<T, double>::value &&
       !std::is_same<T, char>::value &&
       !std::is_same<T, unsigned char>::value) {
        fprintf(stderr, "File: tensor_inpl.h, line: %d. Only digital type allowed in Tensor\n", __LINE__);
        exit(-1);
    }
    data = nullptr;
    mem_addr = nullptr;
    is_num = false;
    cut = 0;
}

template<typename T>
Tensor<T>::Tensor(const std::vector<int>& size) {
    /*
     * Tensor 构造函数：
     * 根据输入size分配空间，设置size
     */
    if(!std::is_same<T, int>::value &&
       !std::is_same<T, unsigned int>::value &&
       !std::is_same<T, long int>::value &&
       !std::is_same<T, unsigned long int>::value &&
       !std::is_same<T, long long int>::value &&
       !std::is_same<T, unsigned long long int>::value &&
       !std::is_same<T, float>::value &&
       !std::is_same<T, double>::value &&
       !std::is_same<T, char>::value &&
       !std::is_same<T, unsigned char>::value) {
        fprintf(stderr, "File: tensor_impl.h, line: %d. Only digital type allowed in Tensor\n", __LINE__);
        exit(-1);
    }
    this->size = size;
    unsigned long space = 1;
    for(const int &i: size) {
        space *= i;
    }
    data = (T*)malloc(sizeof(T)*space);
    mem_addr = data;
    is_num = false;
    cut = 0;

    // 对刚分配的空间引用计数(如果没有bug，应该找不到)
    if(counter.find(mem_addr) != counter.end()) {   // if found
        fprintf(stderr, "File: tensor_impl.h, line: %d. Found just malloced \'data\' in counter "
                        "when constructing\n", __LINE__);
        exit(-1);
    }
    else {
        counter.insert(std::pair<T*, int>(mem_addr, 1));
    }
}

template<typename T>
Tensor<T>::Tensor(const Tensor<T> &src) {
    /*
     * 拷贝构造函数
     */
    data = src.data;
    mem_addr = src.mem_addr;
    is_num = src.is_num;
    size = src.size;
    if(src.cut > 0) {
        cut = src.cut - 1;
    }
    else {
        cut = 0;
    }
    if(counter.find(mem_addr) != counter.end()) {
        counter[mem_addr]++;
    }
    else {
        fprintf(stderr, "File: tensor_impl.h, line: %d. Not found src \'data\' when copy "
                        "constructing\n", __LINE__);
        exit(-1);
    }
}

template<typename T>
Tensor<T>::~Tensor() {
    /*
     * Tensor destructor：
     * 释放内存
     */
    // 查找引用计数
    if(data == nullptr || mem_addr == nullptr) {
        // 如果对象只创建但没分配内存，那么data和mem_addr应该都是null
        if(data != nullptr || mem_addr != nullptr) {
            std::cerr << "Found data and mem_addr only 1 equal to nullptr\n";
        }
        return;
    }
    if(counter.find(mem_addr) != counter.end()) {
        counter[mem_addr]--;
        if(counter[mem_addr] == 0) {
            free(mem_addr);
            counter.erase(mem_addr);
        }
    }
    else {
        fprintf(stderr, "File: tensor_impl.h, line: %d. Not found \'data\' in counter when "
                        "destructing\n", __LINE__);
        exit(-1);
    }
}

template<typename T>
Tensor<T> Tensor<T>::operator[](int index)
{
    /*
     * 重载[]
     * 注：[]不复制内存
     * 创建对象，使data指向截取后地址。mem_addr仍指向旧地址，以保持对整块内存的引用计数
     * 修改size
     */
    Tensor<T> temp;
    /*
     * 例:
     * 当声明3维数组A时，A[0]应返回2维数组，A[0][0]应返回1维数组，A[0][0][0]应返回数值
     * 但由于C++的限制(也可能是我太菜)，无法让一个函数返回两种类型，所以A[0][0][0]仍返回数组，但使用is_num属性标识它已经是
     * 一个数值，并可使用to_num()将这个数值提取出来
     */
    if(is_num) {
        fprintf(stderr, "File: tensor_impl.h, line: %d. You cannot use [] on a num\n", __LINE__);
        exit(-1);
    }
    if(index >= size[0]) {
        fprintf(stderr, "File: tensor_impl.h, line: %d. Index out of range\n", __LINE__);
        exit(-1);
    }
    int other_dim_len = 1;      // 除第一维度外，其他维度长度的乘积，用于计算截取后长度
    for(int i = 1; i<(int)size.size(); i++) {
        other_dim_len *= size[i];
    }
    temp.data = data + index*other_dim_len;
    temp.mem_addr = mem_addr;
    temp.cut = 2;
    if(size.size() == 1) {      // 如果原数组只有1个维度，截取后变为数值
        temp.is_num = true;
        temp.size.push_back(1);
    }
    else {
        for (int i = 1; i < (int)size.size(); i++) {
            temp.size.push_back(size[i]);
        }
    }
    // 增加引用计数
    if(counter.find(mem_addr) != counter.end()) {
        counter[mem_addr]++;
    }
    else {
        fprintf(stderr, "File: tensor_impl.h, line: %d. Not found \'data\' in []\n", __LINE__);
        exit(-1);
    }
    return temp;
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(Tensor<T> array)
{
    /*
     * 重载=
     * 两种情况：
     * 1. 左值是截取：B[i] = A：需要检查左值和右值尺寸一致，并复制数据（deep copy）
     * 2. 左值不是截取：只改变指针。对原地址减引用计数，复制属性，对新地址增引用计数
     */
    if(this->cut > 0) {         // 如果左值是截取
        if(this->size != array.size) {
            fprintf(stderr, "File: tensor_impl.h, line: %d. Could not broadcast input array "
                            "from shape (", __LINE__);
            for(int i = 0; i<(int)array.size.size(); i++) {
                std::cerr << array.size[i] << ", ";
            }
            std::cerr << ") into shape (";
            for(int i = 0; i<(int)this->size.size(); i++) {
                std::cerr << this->size[i] << ", ";
            } 
            std::cerr << ")\n";
            exit(-1);
        }
        int space = 1;
        for(int i = 0; i<(int)array.size.size(); i++) {
            space *= array.size[i];
        }
        memcpy(this->data, array.data, sizeof(T)*space);
    }
    else {                      // 如果左值不是截取
        // 减引用计数
        if(this->mem_addr != nullptr) {
            // 有时只声明对象但没有分配空间，此时不需要减引用计数
            if (counter.find(this->mem_addr) != counter.end()) {
                counter[mem_addr]--;
                if (counter[mem_addr] == 0) {
                    free(mem_addr);
                    counter.erase(mem_addr);
                }
            } else {
                fprintf(stderr, "File: tensor_impl.h, line: %d. Not found old \'data\' in counter "
                                "when operator=\n", __LINE__);
                exit(-1);
            }
        }

        // 复制属性
        this->data = array.data;
        this->mem_addr = array.mem_addr;
        this->size = array.size;
        this->is_num = array.is_num;
        this->cut = 0;

        // 引用计数
        if(counter.find(this->mem_addr) != counter.end()) {
            counter[mem_addr]++;
        }
        else {
            fprintf(stderr, "File: tensor_impl.h, line: %d. Not found new \'data\' in "
                            "counter when operator=\n", __LINE__);
            exit(-1);
        }
    }

    return *this;
}

template<typename T>
T Tensor<T>::to_num() {
    /*
     * 如果数组已经是数值，返回这个值
     */
    if(!is_num) {
        fprintf(stderr, "File: tensor_impl.h, line: %d. You cannot call to_num on a non-num array\n", __LINE__);
        exit(-1);
    }
    return data[0];
}

template<typename T>
Tensor<T> &Tensor<T>::operator=(T value) {
    /*
     * 重载 =，使能对is_num数组直接复制
     * 例如, array[0][2][3] = 5;
     */
    if(!is_num) {
        fprintf(stderr, "File: tensor_impl.h, line: %d. You cannot assign value to a non-num array\n", __LINE__);
        exit(-1);
    }
    data[0] = value;
    return *this;
}

template<typename T>
void Tensor<T>::set_zero() {
    /*
     * data中所有值设为0
     */
    if(data == nullptr) {
        fprintf(stderr, "File: tensor_impl.h, line: %d. You cannot set zero to an array not malloced\n", __LINE__);
        exit(-1);
    }
    int space = 1;
    for(const int &i: size) {
        space *= i;
    }
    memset(data, 0, sizeof(T)*space);
}

template<typename T>
void Tensor<T>::set_rand() {
    /*
     * data中所有值设为随机值
     */
    if(data == nullptr) {
        fprintf(stderr, "File: tensor_impl.h, line: %d. You cannot set rand to an array not malloced\n", __LINE__);
        exit(-1);
    }
    int space = 1;
    for(const int &i: size) {
        space *= i;
    }
    for(int i = 0; i<space; i++) {
        data[i] = ((T)rand()/RAND_MAX) * 2 - 1;
    }
}

template<typename T>
Tensor<T> Tensor<T>::reshape(const std::vector<int> &new_size) {
    /*
     * reshape
     * 检查旧size和新size
     * 创建新对象使其data指向现有data，使其size为新size
     */
    if(is_num) {
        std::cerr << "You cannot reshape a num\n";
    }
    // 检查new_size：负值不能超过1个。不能有0。如果没有负值，则新size对应的内存大小应与原来的相同
    int old_space = 1;
    for(const int &i: size) {
        old_space *= i;
    }
    int negative_count = 0;    // new_size中负值数量
    int new_space = 1;
    for(const int &i: new_size) {
        if(i < 0) {
            negative_count++;
        }
        else if(i == 0) {
            fprintf(stderr, "File: tensor_impl.h, line: %d. Cannot reshape array of "
                            "size %d into shape(", __LINE__, old_space);
            for(const int &j: new_size) {
                fprintf(stderr, "%d, ", j);
            }
            fprintf(stderr, ")\n");
            exit(-1);
        }
        else {
            new_space *= i;
        }
    }
    if(negative_count > 1) {
        fprintf(stderr, "File: tensor_impl.h, line: %d. Can only specify one unknown dimension\n", __LINE__);
        exit(-1);
    }
    if(negative_count == 0) {
        if(old_space != new_space) {
            fprintf(stderr, "File: tensor_impl.h, line: %d. Cannot reshape array of "
                            "size %d into shape(", __LINE__, old_space);
            for(const int &j: new_size) {
                fprintf(stderr, "%d, ", j);
            }
            fprintf(stderr, ")\n");
            exit(-1);
        }
    }
    // 如果size检查通过，对负值进行推断
    Tensor<T> temp;
    temp.data = this->data;
    temp.mem_addr = this->mem_addr;
    temp.size = new_size;
    for(int i = 0; i<(int)new_size.size(); i++) {
        if(temp.size[i] < 0) {
            temp.size[i] = old_space / new_space;
        }
    }
    temp.cut = 0;
    temp.is_num = this->is_num;
    // 引用计数
    if(counter.find(mem_addr) != counter.end()) {
        counter[mem_addr]++;
    }
    else {
        fprintf(stderr, "File: tensor_impl.h, line: %d. Not found \'data\' when reshape\n", __LINE__);
        exit(-1);
    }
    return temp;
}

template<typename T>
void Tensor<T>::print() {
    /*
     * 打印数据
     */
    if(size.size() > 1) {
        for (int i = 0; i < size[0]; i++) {
            (*this)[i].print();
        } printf("\n");
    }
    else {
        for(int i = 0; i<size[0]; i++) {
            if(std::is_same<T, int32>::value) {
                printf("%d ", data[i]);
            }
            else if(std::is_same<T, uint32>::value) {
                printf("%u ", data[i]);
            }
            else if(std::is_same<T, long int>::value) {
                printf("%ld ", data[i]);
            }
            else if(std::is_same<T, unsigned long>::value) {
                printf("%lu ", data[i]);
            }
            else if(std::is_same<T, long long int>::value) {
                printf("%lld ", data[i]);
            }
            else if(std::is_same<T, unsigned long long>::value) {
                printf("%llu ", data[i]);
            }
            else if(std::is_same<T, float32>::value) {
                printf("%f ", data[i]);
            }
            else if(std::is_same<T, float64>::value) {
                printf("%lf ", data[i]);
            }
            else if(std::is_same<T, int8>::value) {
                printf("%d ", data[i]);
            }
            else if(std::is_same<T, uint8>::value) {
                printf("%d ", data[i]);
            }
        } printf("\n");
    }
}

template<typename T>
Tensor<T> Tensor<T>::transpose(const std::vector<int> &new_order) {
    /*
     * transpose:
     * transpose(2,0,1) => dst[k][i][j] = src[i][j][k]
     */
    // new_order的长度应与旧size相同
    if(this->size.size() != new_order.size()) {
        fprintf(stderr, "File: tensor_impl.h, line: %d. Axes don't match array\n", __LINE__);
        exit(-1);
    }
    // 检查new_order：每个元素只能出现一次，且其值应在0到n-1之间
    int new_order_len = (int)new_order.size();
    int appear[new_order_len];
    for(int i = 0; i<new_order_len; i++) {
        appear[i] = 0;
    }
    for(int i = 0; i<new_order_len; i++) {
        if(new_order[i] >= new_order_len || new_order[i] < 0) {
            fprintf(stderr, "File: tensor_impl.h, line: %d. Axis %d is out of bounds "
                            "for array of dimension %d\n", __LINE__, new_order[i], new_order_len);
            exit(-1);
        }
        appear[new_order[i]]++;
    }
    for(int i = 0; i<new_order_len; i++) {
        if(appear[i] > 1) {
            fprintf(stderr, "File: tensor_impl.h, line: %d. Repeated axis in transpose\n", __LINE__);
            exit(-1);
        }
        else if(appear[i] == 0) {
            // Do nothing。由于我们已经检查了new_order的长度和里面的数值，根据抽屉原理，如果有一个元素没出现，那么一定有重复的
            // 而重复的在上面已经处理过了
        }
    }
    // 创建新数组，使其size为transpose之后的size
    std::vector<int> new_size;
    for(const int& i: new_order) {
        new_size.push_back(this->size[i]);
    }
    Tensor<T> temp(new_size);
    // 遍历旧数组中的元素(根据data进行遍历)，计算其下标，并转换到新下标
    int dim = this->size.size();    // 维度
    int old_index[dim];             // 旧下标
    int new_index[dim];             // 新下标
    int array_len = 1;              // 数组中元素个数
    for(const int& i: this->size) {
        array_len *= i;
    }
    for(int i = 0; i<dim; i++) {
        old_index[i] = 0;
    }
    for(int index = 0; index<array_len; index++) {  // 遍历旧数组
        // 旧下标映射到新下标
        for(int i = 0; i<dim; i++) {
            new_index[i] = old_index[new_order[i]];
        }
        // 计算该元素在新数组的位置（位置：相对于data的偏移）
        int offset = 0;
        int weight = 1;
        for(int p = dim-1; p>=0; p--) {
            offset += new_index[p] * weight;
            weight *= new_size[p];
        }
        temp.data[offset] = this->data[index];
        // old_index加一
        array_add_1(old_index, this->size);
    }
    return temp;
}

template<typename T>
std::vector<int> Tensor<T>::shape() {
    /*
     * 返回size
     */
    return size;
}

template<typename T>
int Tensor<T>::argmax() {
    /*
     * argmax
     */
    int len = 1;
    for(const int &i: size) {
        len *= i;
    }
    T max = data[0];
    int index = 0;
    for(int i = 0; i<len; i++) {
        if(data[i] > max) {
            max = data[i];
            index = i;
        }
    }
    return index;
}

template<typename T>
Tensor<int> Tensor<T>::argmax(int axis) {
    /*
     * 在给定axis上argmax
     * 创建新数组，其维度应比旧数组少1
     * 新数组各维度长度应与旧数组除axis指定维度外其他维度一一对应
     * 遍历新数组的每个元素，并在旧数组中axis指定维度上计算argmax
     * 例如：
     * A.size = (2,3,4,5,6)
     * O = A.argmax(2)
     * 那么：
     * O.size = (2,3,5,6)
     * for i in range(2):
     *      for j in range(3):
     *          for k in range(5):
     *              for l in range(6):
     *                  O[i][j][k][l] = max(A[i][j][:][k][l])
     */
    // 检查axis: axis不应超过维度，不应小于0
    if(axis >= (int)size.size() || axis < 0) {
        fprintf(stderr, "File: tensor_impl.h, line: %d. axis %d is out of bounds "
                        "for array of dimension %d\n", __LINE__, axis, (int)size.size());
        exit(-1);
    }
    /*
     * 如果原维度为1，那么返回值应为数值。但此函数无法返回数值，所以返回is_num数组
     */
    if(size.size() == 1) {
        Tensor<int> result(std::vector<int>{1});
        result.data[0] = this->argmax();
        return result;
    }

    // 如果原数组维度大于1
    // 计算结果数组的size
    std::vector<int> new_size;
    for(int i = 0; i<(int)size.size(); i++) {
        if(i == axis) {
            continue;
        }
        else {
            new_size.push_back(size[i]);
        }
    }
    // 使用new_size创建结果数组
    Tensor<int> result{new_size};
    // 遍历结果数组的每个元素，计算其argmax
    int new_dim = (int)new_size.size();
    int old_dim = new_dim+1;
    int new_index[new_dim];     // result中正在访问元素的下标
    int old_index[old_dim];     // 原数组中要访问的元素的下标
    for(int i = 0; i<new_dim; i++) {        // 初始化 new_index 和 old_index
        new_index[i] = 0;
        old_index[i] = 0;
    }
    old_index[old_dim-1] = 0;
    int new_len = 1;                    // result数据空间长度
    for(const int &i: new_size) {
        new_len *= i;
    }
    for(int i = 0; i<new_len; i++) {    // 遍历result
        // 映射new_index 到 old_index
        for(int j = 0; j<old_dim; j++) {
            if(j < axis) {
                old_index[j] = new_index[j];
            }
            else if(j > axis) {
                old_index[j] = new_index[j-1];
            }
        }
        // 设置遍历旧数组axis维的起始下标
        old_index[axis] = 0;
        // 计算元素在旧数组的位置（位置：相对data的offset）
        int offset = 0;
        int weight = 1;
        for(int p = old_dim-1; p>=0; p--) {
            offset += old_index[p] * weight;
            weight *= this->size[p];
        }
        // 初始化 max 和 index
        T max = this->data[offset];
        int index = 0;
        for(int j = 0; j<this->size[axis]; j++) {   // 遍历旧数组axis维元素
            old_index[axis] = j;
            // 计算元素在旧数组的位置（位置：相对data的offset）
            offset = 0;
            weight = 1;
            for(int p = old_dim-1; p>=0; p--) {
                offset += old_index[p] * weight;
                weight *= this->size[p];
            }
            // 比较max，修改index
            if(this->data[offset] > max) {
                max = this->data[offset];
                index = j;
            }
        }
        // 存储index到result
        result.data[i] = index;
        // new_index加一
        array_add_1(new_index, new_size);
    }
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::divide(T divisor) {
    /*
     * 除法: 对true_divide的封装
     * 创建新的数组并分配新的空间。将原数组每个值除以divisor并赋值给新数组
     */
    return this->true_divide(divisor);
}

template<typename T>
Tensor<T> Tensor<T>::true_divide(T divisor) {
    /*
     * 除法
     * 创建新的数组并分配新的空间。将原数组每个值除以divisor并赋值给新数组
     */
    Tensor<T> result{this->size};
    int len = this->len();
    for(int i = 0; i<len; i++) {
        result.data[i] = this->data[i] / divisor;
    }
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::floor_divide(T divisor) {
    /*
     * 向下取整除法
     * 创建新的数组并分配新的空间。将原数组每个值除以divisor并赋值给新数组
     */
    Tensor<T> result{this->size};
    int len = this->len();
    for(int i = 0; i<len; i++) {
        result.data[i] = (int)(this->data[i] / divisor);
    }
    return result;
}

template<typename T>
int Tensor<T>::len() {
    /*
     * 获取当前Tensor数据空间元素数量
     */
    int len = 1;
    for(const int &i: size) {
        len *= i;
    }
    return len;
}

template<typename T>
Tensor<T> Tensor<T>::operator/(T divisor) {
    /*
     * 重载 /
     */
    return this->true_divide(divisor);
}

template<typename T>
Tensor<T> Tensor<T>::broadcast_to(const std::vector<int> &target_size) {
    /*
     * broadcast_to: 将原数组广播为size
     * 1. 让size向target_size看齐，不足的部分在前面加1补齐，得到new_this(即若target=(2,3,4), size=(4), 则new_size=(1,1,4))
     * 2. 返回数组result的size应与target_size相同
     * 3. 如果target_this的某个轴和result的对应轴的长度相同或者其长度为1时，这个数组能够用来计算，否则出错
     * 4. 当new_this的某个轴的长度为1时，沿着此轴运算时都用此轴上的第一组值
     */
    /*
     * 以上为理论算法，实际算法如下：
     * 1. 只新建new_this的new_size，不新建new_this。
     * 2. 按照上面第3条规则检查new_size和target_size
     * 3. 使用new_size新建result
     * 4. 遍历result，按照上面第4条规则从new_this取数据(虽然没有创建new_this，但事实上this扩展为new_this后，data中数据
     *      偏移是不变的，所以可以用new_this的offset从this中取数据)
     */
    // 检查target_size中不能有负值
    for(const int &i: target_size) {
        if(i <= 0) {
            fprintf(stderr, "File: tensor_impl.h, line: %d. all elements of broadcast "
                            "shape must be non-negative\n", __LINE__);
            exit(-1);
        }
    }
    // 检查target_size维度不能比this少
    if(target_size.size() < this->size.size()) {
        fprintf(stderr, "File: tensor_impl.h, line: %d. broadcast shape has less dimensions than now\n", __LINE__);
        exit(-1);
    }
    // 新建new_size
    std::vector<int> new_size;
    int target_dim = (int)target_size.size();
    int this_dim = this->size.size();
    for(int i = 0; i<target_dim-this_dim; i++) {
        new_size.push_back(1);
    }
    for(int i = 0; i<this_dim; i++) {
        new_size.push_back(this->size[i]);
    }
    // 检查new_size和target_size
    for(int i = 0; i<target_dim; i++) {
        if(new_size[i] == target_size[i]) {
            continue;
        }
        else if(new_size[i] == 1) {
            continue;
        }
        else {
            fprintf(stderr, "File: tensor_impl.h, line: %d. cannot broadcast (", __LINE__);
            for(const int &s: this->size) {
                fprintf(stderr, "%d, ", s);
            }
            fprintf(stderr, ") to (");
            for(const int &s: target_size) {
                fprintf(stderr, "%d, ", s);
            }
            fprintf(stderr, ")\n");
            exit(-1);
        }
    }
    // 使用new_size新建result
    Tensor<T> result{target_size};
    // 遍历result，按照上面第4条规则从new_this取数据
    int res_index[target_dim];              // 遍历result时，当前访问元素的index
    int this_index[target_dim];             // result中元素对应的new_this中元素的下标
    memset(res_index, 0, sizeof(T)*target_dim);
    memset(this_index, 0, sizeof(T)*target_dim);
    int target_len = result.len();
    for(int i = 0; i<target_len; i++) { // 遍历result
        // 将res_index映射到this_index
        for(int j = 0; j<target_dim; j++) {
            if(new_size[j] == 1) {
                this_index[j] = 0;
            }
            else {
                this_index[j] = res_index[j];
            }
        }
        // 计算this_index相对于data的偏移
        int offset = 0;
        int weight = 1;
        for(int p = target_dim-1; p>=0; p--) {
            offset += weight * this_index[p];
            weight *= new_size[p];
        }
        // 赋值
        result.data[i] = this->data[offset];
        // res_index加一
        array_add_1(res_index, target_size);
    }
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::divide(Tensor<T> divisor) {
    /*
     * 除法，对true_divide的封装
     */
    return this->true_divide(divisor);
}

template<typename T>
Tensor<T> Tensor<T>::true_divide(Tensor<T> divisor) {
    /*
     * 除法。将this广播到divisor，或将divisor广播到this，然后进行elementwise除法
     */
    Tensor<T> broadcasted_this;
    Tensor<T> broadcasted_divisor;
    if(this->size.size() < divisor.size.size()) {
        broadcasted_this = this->broadcast_to(divisor.size);
        broadcasted_divisor = divisor;
    }
    else {
        broadcasted_this = *this;
        broadcasted_divisor = divisor.broadcast_to(this->size);
    }
    Tensor<T> result{broadcasted_this.size};
    int len = result.len();
    for(int i = 0; i<len; i++) {
        result.data[i] = broadcasted_this.data[i] / broadcasted_divisor.data[i];
    }
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::floor_divide(Tensor<T> divisor) {
    /*
     * 向下取整除法。将this广播到divisor，或将divisor广播到this，然后进行elementwise除法
     */
    Tensor<T> broadcasted_this;
    Tensor<T> broadcasted_divisor;
    if(this->size.size() < divisor.size.size()) {
        broadcasted_this = this->broadcast_to(divisor.size);
        broadcasted_divisor = divisor;
    }
    else {
        broadcasted_this = *this;
        broadcasted_divisor = divisor.broadcast_to(this->size);
    }
    Tensor<T> result{broadcasted_this.size};
    int len = result.len();
    for(int i = 0; i<len; i++) {
        result.data[i] = (int)(broadcasted_this.data[i] / broadcasted_divisor.data[i]);
    }
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator/(Tensor<T> divisor) {
    /*
     * overload /
     */
    return this->true_divide(divisor);
}

template<typename T>
Tensor<float32> Tensor<T>::astype_float32() {
    /*
     * 创建一个新数组，使其值与this相同，但类型为float32
     */
    Tensor<float32> result{this->size};
    int len = result.len();
    for(int i = 0; i<len; i++) {
        result.data[i] = (float32)(this->data[i]);
    }
    return result;
}

template<typename T>
Tensor<int32> Tensor<T>::astype_int32() {
    /*
     * 创建一个新数组，使其值与this相同，但类型为int32
     */
    Tensor<int32> result{this->size};
    int len = result.len();
    for(int i = 0; i<len; i++) {
        result.data[i] = (int32)(this->data[i]);
    }
    return result;
}

template<typename T>
Tensor<uint8> Tensor<T>::astype_uint8() {
    /*
     * 创建一个新数组，使其值与this相同，但类型为uint8
     */
    Tensor<uint8> result{this->size};
    int len = result.len();
    for(int i = 0; i<len; i++) {
        result.data[i] = (uint8) (this->data[i]);
    }
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::deep_copy() {
    /*
     * 创建一个新的数组，使其与this完全相同(但拥有不同的内存空间)，然后返回它
     */
    Tensor<T> ret{this->size};
    int len = ret.len();
    memcpy(ret.data, this->data, sizeof(T)*len);
    return ret;
}


//template<typename T>
//Tensor<T> Tensor<T>::dot(Tensor<T> B_tensor) {
//    /*
//     * 矩阵乘法。原始算法
//     */
//    Tensor<T> A_tensor = *this;
//    Tensor<T> C_tensor{std::vector<int>{A_tensor.size[0], B_tensor.size[1]}};
//    T * A = A_tensor.data;
//    T * B = B_tensor.data;
//    T * C = C_tensor.data;
//    int M = A_tensor.size[0];
//    int K = A_tensor.size[1];
//    int N = B_tensor.size[1];
//
//    for(int i = 0; i<M; i++) {
//        for(int j = 0; j<N; j++) {
//            T temp = 0;
//            for(int k = 0; k<K; k++) {
//                temp += A[i * K + k] * B[k * N + j];
//            }
//            C[i * N + j] = temp;
//        }
//    }
//
//    return C_tensor;
//}

//template<typename T>
//Tensor<T> Tensor<T>::dot(Tensor<T> B_tensor) {
//    /*
//     * 矩阵乘法。利用cache line，在矩阵B取数时，同时使用在同一个cache line中的数据进行计算
//     * 即在计算时，一次计算C的横向相连的4个数据，使用A的1行和B的4列
//     */
//    Tensor<T> A_tensor = *this;
//    Tensor<T> C_tensor{std::vector<int>{A_tensor.size[0], B_tensor.size[1]}};
//    T * A = A_tensor.data;
//    T * B = B_tensor.data;
//    T * C = C_tensor.data;
//    int M = A_tensor.size[0];
//    int K = A_tensor.size[1];
//    int N = B_tensor.size[1];
//
//    for(int i = 0; i<M; i++) {
//        int new_N = N / 4 * 4;
//        for(int j = 0; j<new_N; j+=4) {
//            T temp1 = 0;
//            T temp2 = 0;
//            T temp3 = 0;
//            T temp4 = 0;
//            for(int k = 0; k<K; k++) {
//                temp1 += A[i*K + k] * B[k*N + j];
//                temp2 += A[i*K + k] * B[k*N + j+1];
//                temp3 += A[i*K + k] * B[k*N + j+2];
//                temp4 += A[i*K + k] * B[k*N + j+3];
//            }
//            C[i*N + j] = temp1;
//            C[i*N + j+1] = temp2;
//            C[i*N + j+2] = temp3;
//            C[i*N + j+3] = temp4;
//        }
//        for(int j = new_N; j<N; j++) {
//            T temp = 0;
//            for(int k = 0; k<K; k++) {
//                temp += A[i*K + k] * B[k*N + j];
//            }
//            C[i*N + j] = temp;
//        }
//    }
//
//    return C_tensor;
//}


//template<typename T>
//Tensor<T> Tensor<T>::dot(Tensor<T> B_tensor) {
//    /*
//     * 矩阵乘法。利用cache line，在矩阵A和B取数时，同时使用在同一个cache line中的数据进行计算
//     * 即在计算时，一次计算C的横向纵向相连的16个数据，使用A的4行和B的4列
//     */
//    Tensor<T> A_tensor = *this;
//    Tensor<T> C_tensor{std::vector<int>{A_tensor.size[0], B_tensor.size[1]}};
//    T * A = A_tensor.data;
//    T * B = B_tensor.data;
//    T * C = C_tensor.data;
//    int M = A_tensor.size[0];
//    int K = A_tensor.size[1];
//    int N = B_tensor.size[1];
//
//    int new_M = M / 4 * 4;
//    for (int i = 0; i < new_M; i+=4) {
//        int new_N = N / 4 * 4;
//        for (int j = 0; j < new_N; j += 4) {
//            float temp11 = 0; float temp12 = 0; float temp13 = 0; float temp14 = 0;
//            float temp21 = 0; float temp22 = 0; float temp23 = 0; float temp24 = 0;
//            float temp31 = 0; float temp32 = 0; float temp33 = 0; float temp34 = 0;
//            float temp41 = 0; float temp42 = 0; float temp43 = 0; float temp44 = 0;
//            for (int k = 0; k < K; k++) {
//                temp11 += A[i * K + k] * B[k * N + j];
//                temp12 += A[(i+1) * K + k] * B[k * N + j];
//                temp13 += A[(i+2) * K + k] * B[k * N + j];
//                temp14 += A[(i+3) * K + k] * B[k * N + j];
//                temp21 += A[i * K + k] * B[k * N + j + 1];
//                temp22 += A[(i+1) * K + k] * B[k * N + j + 1];
//                temp23 += A[(i+2) * K + k] * B[k * N + j + 1];
//                temp24 += A[(i+3) * K + k] * B[k * N + j + 1];
//                temp31 += A[i * K + k] * B[k * N + j + 2];
//                temp32 += A[(i+1) * K + k] * B[k * N + j + 2];
//                temp33 += A[(i+2) * K + k] * B[k * N + j + 2];
//                temp34 += A[(i+3) * K + k] * B[k * N + j + 2];
//                temp41 += A[i * K + k] * B[k * N + j + 3];
//                temp42 += A[(i+1) * K + k] * B[k * N + j + 3];
//                temp43 += A[(i+2) * K + k] * B[k * N + j + 3];
//                temp44 += A[(i+3) * K + k] * B[k * N + j + 3];
//            }
//            C[i * N + j] = temp11;
//            C[(i+1) * N + j] = temp12;
//            C[(i+2) * N + j] = temp13;
//            C[(i+3) * N + j] = temp14;
//            C[i * N + j + 1] = temp21;
//            C[(i+1) * N + j + 1] = temp22;
//            C[(i+2) * N + j + 1] = temp23;
//            C[(i+3) * N + j + 1] = temp24;
//            C[i * N + j + 2] = temp31;
//            C[(i+1) * N + j + 2] = temp32;
//            C[(i+2) * N + j + 2] = temp33;
//            C[(i+3) * N + j + 2] = temp34;
//            C[i * N + j + 3] = temp41;
//            C[(i+1) * N + j + 3] = temp42;
//            C[(i+2) * N + j + 3] = temp43;
//            C[(i+3) * N + j + 3] = temp44;
//        }
//        for (int j = new_N; j < N; j++) {
//            float temp1 = 0; float temp2 = 0; float temp3 = 0; float temp4 = 0;
//            for (int k = 0; k < K; k++) {
//                temp1 += A[i * K + k] * B[k * N + j];
//                temp2 += A[(i+1) * K + k] * B[k * N + j];
//                temp3 += A[(i+2) * K + k] * B[k * N + j];
//                temp4 += A[(i+3) * K + k] * B[k * N + j];
//            }
//            C[i * N + j] = temp1;
//            C[(i+1) * N + j] = temp2;
//            C[(i+2) * N + j] = temp3;
//            C[(i+3) * N + j] = temp4;
//        }
//    }
//    for (int i = new_M; i < M; i++) {
//        int new_N = N / 4 * 4;
//        for (int j = 0; j < new_N; j += 4) {
//            float temp1 = 0;
//            float temp2 = 0;
//            float temp3 = 0;
//            float temp4 = 0;
//            for (int k = 0; k < K; k++) {
//                temp1 += A[i * K + k] * B[k * N + j];
//                temp2 += A[i * K + k] * B[k * N + j + 1];
//                temp3 += A[i * K + k] * B[k * N + j + 2];
//                temp4 += A[i * K + k] * B[k * N + j + 3];
//            }
//            C[i * N + j] = temp1;
//            C[i * N + j + 1] = temp2;
//            C[i * N + j + 2] = temp3;
//            C[i * N + j + 3] = temp4;
//        }
//        for (int j = new_N; j < N; j++) {
//            float temp = 0;
//            for (int k = 0; k < K; k++) {
//                temp += A[i * K + k] * B[k * N + j];
//            }
//            C[i * N + j] = temp;
//        }
//    }
//
//    return C_tensor;
//}


template<typename T>
Tensor<T> Tensor<T>::dot(Tensor<T> B_tensor) {
    /*
     * 矩阵乘法。多线程
     */
    Tensor<T> A_tensor = *this;
    Tensor<T> C_tensor{std::vector<int>{A_tensor.size[0], B_tensor.size[1]}};
    T * A = A_tensor.data;
    T * B = B_tensor.data;
    T * C = C_tensor.data;
    int M = A_tensor.size[0];
    int K = A_tensor.size[1];
    int N = B_tensor.size[1];

    // 最大线程数n_proc。每150000计算量增加一个线程，最大不超过n_proc
    int n_proc = (int)sysconf(_SC_NPROCESSORS_ONLN);
    int max_calc_amount = 150000 * n_proc;
    n_proc = n_proc - (max_calc_amount - M*K*N) / 150000;

    int M_per_proc = M / n_proc;
    std::thread t[n_proc];
    for(int i = 0; i<n_proc; i++) {
        t[i] = std::thread(mt_dot, &C[i*M_per_proc*N], &A[i*M_per_proc*K], B, M_per_proc, K, N);
    }
    mt_dot(&C[n_proc*M_per_proc*N], &A[n_proc*M_per_proc*K], B, M-n_proc*M_per_proc, K, N);
    for(int i = 0; i<n_proc; i++) {
        t[i].join();
    }

    return C_tensor;
}


template<typename T>
Tensor<T> Tensor<T>::add(T adder) {
    /*
     * add
     */
    Tensor<T> result{this->size};
    int len = result.len();
    for(int i = 0; i<len; i++) {
        result.data[i] = this->data[i] + adder;
    }
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator+(T adder) {
    /*
     * overload +
     */
    return this->add(adder);
}

template<typename T>
Tensor<T> &Tensor<T>::operator+=(T adder) {
    /*
     * overload +=
     */
    int len = this->len();
    for(int i = 0; i<len; i++) {
        this->data[i] += adder;
    }
    return *this;
}

template<typename T>
Tensor<T> Tensor<T>::add(Tensor<T> adder) {
    /*
     * 加法。将this广播到adder，或将adder广播到this，然后进行elementwise加法
     */
    Tensor<T> broadcasted_this;
    Tensor<T> broadcasted_adder;
    if(this->size.size() < adder.size.size()) {
        broadcasted_this = this->broadcast_to(adder.size);
        broadcasted_adder = adder;
    }
    else {
        broadcasted_this = *this;
        broadcasted_adder = adder.broadcast_to(this->size);
    }
    Tensor<T> result{broadcasted_this.size};
    int len = result.len();
    for(int i = 0; i<len; i++) {
        result.data[i] = broadcasted_this.data[i] + broadcasted_adder.data[i];
    }
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator+(Tensor<T> adder) {
    /*
     * overload +
     */
    return this->add(adder);
}

template<typename T>
Tensor<T> Tensor<T>::concat(Tensor<T> array, int dim) {
    /*
     * concat: 创建新张量使其尺寸为拼接后的尺寸。遍历新张量的元素，根据其偏移计算下标，根据下标
     * 计算应从哪个输入张量中提取元素，并计算其下标，根据此下标计算该元素在输入张量中的偏移，并使用
     * 此元素填充新张量
     */
    // 计算新尺寸(检查：拼接双方维度相同，dim外其他维度尺寸相同)
    std::vector<int> new_size;
    if(this->size.size() != array.size.size()) {
        fprintf(stderr, "the dimension of the tensors to concat should be same\n");
        exit(-1);
    }
    for(int i = 0; i<(int)this->size.size(); i++) {
        if(i != dim) {
            if(this->size[i] != array.size[i]) {
                fprintf(stderr, "the dimensions to concat should exactly match. Cannot match"
                                "%d to %d\n", this->size[i], array.size[i]);
                exit(-1);
            }
            new_size.push_back(this->size[i]);
        }
        else {
            new_size.push_back(this->size[i] + array.size[i]);
        }
    }
    // 创建返回对象
    Tensor<T> result{new_size};
    // concat
    int new_dim = (int)new_size.size();
    int new_index[new_dim];
    int old_index[new_dim];
    memset(new_index, 0, sizeof(int)*new_dim);
    memset(old_index, 0, sizeof(int)*new_dim);
    int len = result.len();
    for(int i = 0; i<len; i++) {    // 遍历新张量中的每个元素
        // new_index即为当前访问元素在result中的下标，判断此下标应从哪个输入张量中取数
        int get_from_this = 0;
        if(new_index[dim] <= this->size[dim]-1) {
            get_from_this = 1;
        }
        // 计算要取的数在输入张量中的下标
        for(int j = 0; j<new_dim; j++) {
            if(j == dim) {
                if(get_from_this) {
                    old_index[j] = new_index[j];
                }
                else {
                    old_index[j] = new_index[j] - this->size[j];
                }
            }
            else {
                old_index[j] = new_index[j];
            }
        }
        // 根据下标计算此数在输入张量中的偏移
        int offset = 0;
        int weight = 1;
        if(get_from_this) {
            for(int j = this->size.size()-1; j >= 0; j--) {
                offset += weight * old_index[j];
                weight *= this->size[j];
            }
        }
        else {
            for(int j = array.size.size()-1; j >= 0; j--) {
                offset += weight * old_index[j];
                weight *= array.size[j];
            }
        }
        // 从输入张量中取数置于result中
        if(get_from_this) {
            result.data[i] = this->data[offset];
        }
        else {
            result.data[i] = array.data[offset];
        }
        // 将新张量下标加一
        array_add_1(new_index, result.size);
    }
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::sub(T subtractend) {
    /*
     * 减法
     */
    Tensor<T> result{this->size};
    int len = this->len();
    for(int i = 0; i<len; i++) {
        result.data[i] = this->data[i] - subtractend;
    }
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator-(T subtractend) {
    /*
     * overload -
     */
    return this->sub(subtractend);
}

template<typename T>
Tensor<T> Tensor<T>::sub(Tensor<T> subtractend) {
    /*
     * 减法。将this广播到adder，或将adder广播到this，然后进行elementwise减法
     */
    Tensor<T> broadcasted_this;
    Tensor<T> broadcasted_subtractend;
    if(this->size.size() < subtractend.size.size()) {
        broadcasted_this = this->broadcast_to(subtractend.size);
        broadcasted_subtractend = subtractend;
    }
    else {
        broadcasted_this = *this;
        broadcasted_subtractend = subtractend.broadcast_to(this->size);
    }
    Tensor<T> result{broadcasted_this.size};
    int len = result.len();
    for(int i = 0; i<len; i++) {
        result.data[i] = broadcasted_this.data[i] - broadcasted_subtractend.data[i];
    }
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator-(Tensor<T> subtractend) {
    /*
     * overload -
     */
    return this->sub(subtractend);
}

template<typename T>
Tensor<T> &Tensor<T>::operator-=(T subtractend) {
    /*
     * overload -=
     */
    int len = this->len();
    for(int i = 0; i<len; i++) {
        this->data[i] -= subtractend;
    }
    return *this;
}

template<typename T>
Tensor<T> Tensor<T>::elewise_sqrt() {
    /*
     * sqrt
     */
    Tensor<T> result{this->size};
    int len = result.len();
    for(int i = 0; i<len; i++) {
        result.data[i] = sqrt(this->data[i]);
    }
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::mult(T multiplier) {
    /*
     * 乘法
     */
    Tensor<T> result{this->size};
    int len = result.len();
    for(int i = 0; i<len; i++) {
        result.data[i] = this->data[i] * multiplier;
    }
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::mult(Tensor<T> multiplier) {
    /*
     * 乘法。将this广播到multiplier，或将multiplier广播到this，然后进行elementwise加法
     */
    Tensor<T> broadcasted_this;
    Tensor<T> broadcasted_multiplier;
    if(this->size.size() < multiplier.size.size()) {
        broadcasted_this = this->broadcast_to(multiplier.size);
        broadcasted_multiplier = multiplier;
    }
    else {
        broadcasted_this = *this;
        broadcasted_multiplier = multiplier.broadcast_to(this->size);
    }
    Tensor<T> result{broadcasted_this.size};
    int len = result.len();
    for(int i = 0; i<len; i++) {
        result.data[i] = broadcasted_this.data[i] * broadcasted_multiplier.data[i];
    }
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator*(T multiplier) {
    /*
     * overload *
     */
    return this->mult(multiplier);
}

template<typename T>
Tensor<T> Tensor<T>::operator*(Tensor<T> multiplier) {
    /*
     * overload *
     */
    return this->mult(multiplier);
}

template<typename T>
Tensor<T> &Tensor<T>::operator*=(Tensor<T> multiplier) {
    /*
     * overload *=
     */
    int len = this->len();
    for(int i = 0; i<len; i++) {
        this->data[i] *= multiplier;
    }
    return *this;
}

template<typename T>
Tensor<T> &Tensor<T>::operator/=(Tensor<T> divisor) {
    /*
     * overload /=
     */
    int len = this->len();
    for(int i = 0; i<len; i++) {
        this->data[i] /= divisor;
    }
    return *this;
}

template<typename T>
Tensor<T> Tensor<T>::mean(std::vector<int> axis) {
    /*
     * mean。对axis指定的轴求平均值
     * 使用不在axis中包含的维度的尺寸创建新数组。遍历新数组，计算其在旧数组中对应的下标，并使用对应的轴
     */
    // 检查输入的axis。不能有重复，不能有负数，轴编号必须比维度数小
    int appear[this->size.size()];
    memset(appear, 0, sizeof(int)*this->size.size());
    for(int &i: axis) {
        if(i < 0) {
            fprintf(stderr, "Negative value not allowed in axis\n");
            exit(-1);
        }
        else if(i >= (int)this->size.size()) {
            fprintf(stderr, "Axis %d is out of bounds for array of dimension of %d\n", i, (int)this->size.size());
            exit(-1);
        }
        else {
            appear[i]++;
        }
    }
    for(int i = 0; i<(int)this->size.size(); i++) {
        if(appear[i] > 1) {
            fprintf(stderr, "duplicate value in axis\n");
            exit(-1);
        }
    }
    if(axis.size() == this->size.size()) {
        Tensor<T> result{std::vector<int>{1}};
        result.data[0] = this->mean();
        result.is_num = 1;
        return result;
    }
    // 计算axis指定轴的尺寸和结果tensor的尺寸   // 考虑到axis中的轴编号可能不按顺序，我们使用之前统计的appear计算
    std::vector<int> new_size;
    std::vector<int> axis_size;
    // 分别使用axis中出现和未出现的轴，按序排放创建vector，用于在遍历新数组时，将新数组坐标映射到旧数组坐标
    std::vector<int> did_appear;
    std::vector<int> never_appear;
    for(int i = 0; i<(int)this->size.size(); i++) {
        if(appear[i]) {
            axis_size.push_back(this->size[i]);
            did_appear.push_back(i);
        }
        else {
            new_size.push_back(this->size[i]);
            never_appear.push_back(i);
        }
    }
    // 创建新tensor
    Tensor<T> result{new_size};
    // 遍历新数组，对于每一个元素，计算axis中指定轴上的平均值
    int len = result.len();
    int new_dim = result.size.size();
    int old_dim = this->size.size();
    int new_index[new_dim];
    int old_index[old_dim];
    memset(new_index, 0, sizeof(int)*new_dim);
    memset(old_index, 0, sizeof(int)*old_dim);
    int axis_dim = (int)axis.size();                    // 指定的轴的个数
    int axis_index[axis_dim];                           // 遍历axis指定轴时，指定轴之间的下标
    int axis_len = this->len() / result.len();          // axis指定轴的元素个数
    for(int i = 0; i<len; i++) {
        // 将新数组下标映射到旧数组下标
        for(int j = 0; j<new_dim; j++) {
            old_index[never_appear[j]] = new_index[j];
        }
        // 对于axis指定的轴，将其提取出来，建立一个下标axis_index
        memset(axis_index, 0, sizeof(int)*axis_dim);
        // 遍历axis_index并求平均值
        T sum = 0;
        for(int j = 0; j<axis_len; j++) {
            // 将axis_index映射到old_index
            for(int k = 0; k<axis_dim; k++) {
                old_index[did_appear[k]] = axis_index[k];
            }
            // 根据old_index求该值在this中的偏移
            int offset = 0;
            int weight = 1;
            for(int p = old_dim-1; p >= 0; p--) {
                offset += weight * old_index[p];
                weight *= this->size[p];
            }
            // 对数值进行累加
            sum += this->data[offset];
            // axis_index加一
            array_add_1(axis_index, axis_size);
        }
        sum /= axis_len;
        // 将平均值存入结果数组
        result.data[i] = sum;
        // 新数组下标加一
        array_add_1(new_index, new_size);
    }
    return result;
}

template<typename T>
T Tensor<T>::mean() {
    /*
     * mean
     */
    T sum = 0;
    int len = this->len();
    for(int i = 0; i<len; i++) {
        sum += this->data[i];
    }
    sum /= len;
    return sum;
}

template<typename T>
T Tensor<T>::var() {
    /*
     * var
     */
    T mean = this->mean();
    T sum = 0;
    int len = this->len();
    for(int i = 0; i<len; i++) {
        sum += (this->data[i] - mean) * (this->data[i] - mean);
    }
    sum /= len;
    return sum;
}

template<typename T>
Tensor<T> Tensor<T>::var(std::vector<int> axis) {
    /*
     * var
     */
    // 首先计算mean
    Tensor<T> mean = this->mean(axis);
    // 检查输入的axis。不能有重复，不能有负数，轴编号必须比维度数小
    // 由于mean中已经检查过axis了，所以这里不再检查
    int appear[this->size.size()];
    memset(appear, 0, sizeof(int)*this->size.size());
    for(int &i: axis) {
        appear[i]++;
    }
    // 如果对所有轴求var，则直接调用var()
    if(axis.size() == this->size.size()) {
        Tensor<T> result{std::vector<int>{1}};
        result.data[0] = this->var();
        result.is_num = 1;
        return result;
    }
    // 计算axis指定轴的尺寸和结果tensor的尺寸   // 考虑到axis中的轴编号可能不按顺序，我们使用之前统计的appear计算
    std::vector<int> new_size;
    std::vector<int> axis_size;
    // 分别使用axis中出现和未出现的轴，按序排放创建vector，用于在遍历新数组时，将新数组坐标映射到旧数组坐标
    std::vector<int> did_appear;
    std::vector<int> never_appear;
    for(int i = 0; i<(int)this->size.size(); i++) {
        if(appear[i]) {
            axis_size.push_back(this->size[i]);
            did_appear.push_back(i);
        }
        else {
            new_size.push_back(this->size[i]);
            never_appear.push_back(i);
        }
    }
    // 创建新tensor
    Tensor<T> result{new_size};
    // 遍历新数组，对于每一个元素，计算axis中指定轴上的平均值
    int len = result.len();
    int new_dim = result.size.size();
    int old_dim = this->size.size();
    int new_index[new_dim];
    int old_index[old_dim];
    memset(new_index, 0, sizeof(int)*new_dim);
    memset(old_index, 0, sizeof(int)*old_dim);
    int axis_dim = (int)axis.size();                    // 指定的轴的个数
    int axis_index[axis_dim];                           // 遍历axis指定轴时，指定轴之间的下标
    int axis_len = this->len() / result.len();          // axis指定轴的元素个数
    for(int i = 0; i<len; i++) {
        // 将新数组下标映射到旧数组下标
        for(int j = 0; j<new_dim; j++) {
            old_index[never_appear[j]] = new_index[j];
        }
        // 对于axis指定的轴，将其提取出来，建立一个下标axis_index
        memset(axis_index, 0, sizeof(int)*axis_dim);
        // 遍历axis_index并求平均值
        T sum = 0;
        for(int j = 0; j<axis_len; j++) {
            // 将axis_index映射到old_index
            for(int k = 0; k<axis_dim; k++) {
                old_index[did_appear[k]] = axis_index[k];
            }
            // 根据old_index求该值在this中的偏移
            int offset = 0;
            int weight = 1;
            for(int p = old_dim-1; p >= 0; p--) {
                offset += weight * old_index[p];
                weight *= this->size[p];
            }
            // 对数值进行累加
            sum += (this->data[offset] - mean.data[i]) * (this->data[offset] - mean.data[i]);
            // axis_index加一
            array_add_1(axis_index, axis_size);
        }
        sum /= axis_len;
        // 将平均值存入结果数组
        result.data[i] = sum;
        // 新数组下标加一
        array_add_1(new_index, new_size);
    }
    return result;
}


#endif //QUANT_TENSOR_IMPL_H
