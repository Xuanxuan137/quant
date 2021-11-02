//
// Created by noname on 2021/10/25.
//

#ifndef QUANT_NDARRAY_IMPL_H
#define QUANT_NDARRAY_IMPL_H

#include "vdarray.h"

template<typename T>
Vdarray<T>::Vdarray() {
    /*
     * Vdarray 构造函数：
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
        std::cerr << "Only digital type allowed in Vdarray\n";
        exit(-1);
    }
    data = nullptr;
    mem_addr = nullptr;
    is_num = false;
    cut = 0;
}

template<typename T>
Vdarray<T>::Vdarray(const std::vector<int>& size) {
    /*
     * Vdarray 构造函数：
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
        std::cerr << "Only digital type allowed in Vdarray\n";
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
        std::cerr << "Found just malloced \'data\' in counter when constructing\n";
        exit(-1);
    }
    else {
        counter.insert(std::pair<T*, int>(mem_addr, 1));
    }
}

template<typename T>
Vdarray<T>::Vdarray(const Vdarray<T> &src) {
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
        std::cerr << "Not found src \'data\' when copy constructing\n";
        exit(-1);
    }
}

template<typename T>
Vdarray<T>::~Vdarray() {
    /*
     * Vdarray destructor：
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
        std::cerr << "Not found \'data\' in counter when destructing\n";
        exit(-1);
    }
}

template<typename T>
Vdarray<T> Vdarray<T>::operator[](int index)
{
    /*
     * 重载[]
     * 注：[]不复制内存
     * 创建对象，使data指向截取后地址。mem_addr仍指向旧地址，以保持对整块内存的引用计数
     * 修改size
     */
    Vdarray<T> temp;
    /*
     * 例:
     * 当声明3维数组A时，A[0]应返回2维数组，A[0][0]应返回1维数组，A[0][0][0]应返回数值
     * 但由于C++的限制(也可能是我太菜)，无法让一个函数返回两种类型，所以A[0][0][0]仍返回数组，但使用is_num属性标识它已经是
     * 一个数值，并可使用to_num()将这个数值提取出来
     */
    if(is_num) {
        std::cerr << "You cannot use [] on a num\n";
        exit(-1);
    }
    if(index >= size[0]) {
        std::cerr << "Index out of range\n";
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
        std::cerr << "Not found \'data\' in []\n";
        exit(-1);
    }
    return temp;
}

template<typename T>
Vdarray<T>& Vdarray<T>::operator=(Vdarray<T> array)
{
    /*
     * 重载=
     * 两种情况：
     * 1. 左值是截取：B[i] = A：需要检查左值和右值尺寸一致，并复制数据（deep copy）
     * 2. 左值不是截取：只改变指针。对原地址减引用计数，复制属性，对新地址增引用计数
     */
    if(this->cut > 0) {         // 如果左值是截取
        if(this->size != array.size) {
            std::cerr << "could not broadcast input array from shape (";
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
                std::cerr << "Not found old \'data\' in counter when operator=\n";
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
            std::cerr << "Not found new \'data\' in counter when operator=\n";
            exit(-1);
        }
    }

    return *this;
}

template<typename T>
T Vdarray<T>::to_num() {
    /*
     * 如果数组已经是数值，返回这个值
     */
    if(!is_num) {
        std::cerr << "You cannot call to_num on a non-num array\n";
        exit(-1);
    }
    return data[0];
}

template<typename T>
Vdarray<T> &Vdarray<T>::operator=(T value) {
    /*
     * 重载 =，使能对is_num数组直接复制
     * 例如, array[0][2][3] = 5;
     */
    if(!is_num) {
        std::cerr << "You cannot assign value to a non-num array\n";
        exit(-1);
    }
    data[0] = value;
    return *this;
}

template<typename T>
void Vdarray<T>::set_zero() {
    /*
     * data中所有值设为0
     */
    if(data == nullptr) {
        std::cerr << "You cannot set zero to an array not malloced\n";
        exit(-1);
    }
    int space = 1;
    for(const int &i: size) {
        space *= i;
    }
    memset(data, 0, sizeof(T)*space);
}

template<typename T>
void Vdarray<T>::set_rand() {
    /*
     * data中所有值设为随机值
     */
    if(data == nullptr) {
        std::cerr << "You cannot set rand to an array not malloced\n";
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
Vdarray<T> Vdarray<T>::reshape(const std::vector<int> &new_size) {
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
            fprintf(stderr, "Cannot reshape array of size %d into shape(", old_space);
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
        fprintf(stderr, "Can only specify one unknown dimension\n");
        exit(-1);
    }
    if(negative_count == 0) {
        if(old_space != new_space) {
            fprintf(stderr, "Cannot reshape array of size %d into shape(", old_space);
            for(const int &j: new_size) {
                fprintf(stderr, "%d, ", j);
            }
            fprintf(stderr, ")\n");
            exit(-1);
        }
    }
    // 如果size检查通过，对负值进行推断
    Vdarray<T> temp;
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
        std::cerr << "Not found \'data\' when reshape\n";
        exit(-1);
    }
    return temp;
}

template<typename T>
void Vdarray<T>::print() {
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
Vdarray<T> Vdarray<T>::transpose(const std::vector<int> &new_order) {
    /*
     * transpose:
     * transpose(2,0,1) => dst[k][i][j] = src[i][j][k]
     */
    // new_order的长度应与旧size相同
    if(this->size.size() != new_order.size()) {
        std::cerr << "Axes don't match array\n";
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
            fprintf(stderr, "Axis %d is out of bounds for array of dimension %d\n", new_order[i], new_order_len);
            exit(-1);
        }
        appear[new_order[i]]++;
    }
    for(int i = 0; i<new_order_len; i++) {
        if(appear[i] > 1) {
            fprintf(stderr, "Repeated axis in transpose\n");
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
    Vdarray<T> temp(new_size);
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
std::vector<int> Vdarray<T>::shape() {
    /*
     * 返回size
     */
    return size;
}

template<typename T>
int Vdarray<T>::argmax() {
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
Vdarray<int> Vdarray<T>::argmax(int axis) {
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
    if(axis >= size.size() || axis < 0) {
        fprintf(stderr, "axis %d is out of bounds for array of dimension %d\n", axis, size.size());
        exit(-1);
    }
    /*
     * 如果原维度为1，那么返回值应为数值。但此函数无法返回数值，所以返回is_num数组
     */
    if(size.size() == 1) {
        Vdarray<int> result(std::vector<int>{1});
        result.data[0] = this->argmax();
        return result;
    }

    // 如果原数组维度大于1
    // 计算结果数组的size
    std::vector<int> new_size;
    for(int i = 0; i<size.size(); i++) {
        if(i == axis) {
            continue;
        }
        else {
            new_size.push_back(size[i]);
        }
    }
    // 使用new_size创建结果数组
    Vdarray<int> result{new_size};
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
Vdarray<T> Vdarray<T>::divide(float64 divisor) {
    /*
     * 除法: 对true_divide的封装
     * 创建新的数组并分配新的空间。将原数组每个值除以divisor并赋值给新数组
     */
    return this->true_divide(divisor);
}

template<typename T>
Vdarray<T> Vdarray<T>::true_divide(float64 divisor) {
    /*
     * 除法
     * 创建新的数组并分配新的空间。将原数组每个值除以divisor并赋值给新数组
     */
    Vdarray<T> result{this->size};
    int len = this->len();
    for(int i = 0; i<len; i++) {
        result.data[i] = this->data[i] / divisor;
    }
    return result;
}

template<typename T>
Vdarray<T> Vdarray<T>::floor_divide(float64 divisor) {
    /*
     * 向下取整除法
     * 创建新的数组并分配新的空间。将原数组每个值除以divisor并赋值给新数组
     */
    Vdarray<T> result{this->size};
    int len = this->len();
    for(int i = 0; i<len; i++) {
        result.data[i] = (int)(this->data[i] / divisor);
    }
    return result;
}

template<typename T>
int Vdarray<T>::len() {
    /*
     * 获取当前Vdarray数据空间元素数量
     */
    int len = 1;
    for(const int &i: size) {
        len *= i;
    }
    return len;
}

template<typename T>
Vdarray<T> Vdarray<T>::operator/(float64 divisor) {
    /*
     * 重载 /
     */
    return this->true_divide(divisor);
}

template<typename T>
Vdarray<T> Vdarray<T>::broadcast_to(const std::vector<int> &target_size) {
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
            fprintf(stderr, "cannot broadcast (");
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
    Vdarray<T> result{target_size};
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


void array_add_1(int array[], const std::vector<int> &size)
{
    /*
     * 给出一个数组array，和数组每一位的进制size。让数组自增一
     * 常用于Vdarray按下标遍历。调用此函数可将下标加一
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
    for(const int &i: size) {
        printf("%d ", i);
    }
    printf("\n");
}


#endif //QUANT_NDARRAY_IMPL_H
