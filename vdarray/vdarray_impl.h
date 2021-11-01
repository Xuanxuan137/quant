//
// Created by noname on 2021/10/25.
//

#ifndef QUANT_NDARRAY_IMPL_H
#define QUANT_NDARRAY_IMPL_H

#include "vdarray.h"

template<typename T>
Vdarray<T>::Vdarray() {
    /*
     * Vdarray constructor：
     * Only create object and do not allocate memory
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
     * Vdarray constructor：
     * Allocate memory according to input size, and set 'size' of Vdarray object
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

    // Reference count the just allocated data(if there is no bug, 'data' cannot be found in reference counter now)
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
     * copy constructor
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
     * release memory
     */
    // search reference counter
    if(data == nullptr || mem_addr == nullptr) {
        // if a Vdarray object is created but never allocated memory for its 'data', the 'data' and 'mem_addr'
        // should both be nullptr
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
     * Overload operator[]
     * Create a new object, and 'data' point to the memory address after interception
     * 'mem_addr' still point to old address, in order to keep reference count to the whole block of memory
     * set size to new size
     */
    Vdarray<T> temp;
    /*
     * Example:
     * When declaring a 3-dimensional array A, A[0] should return a 2-dim array, A[0][0] should return a 1-dim array,
     * and A[0][0][0] should return a num.
     * But since the limitation of the explicit data type of C++(or maybe since I'm vegetable),
     * we cannot let [] return both an array and a num, so when [] should return a num, it still returns an array,
     * an array which has only one data in 'data', and set is_num to true at the same time, indicate that this
     * array is already a num, and to_num() is can be used to get this num.
     */
    if(is_num) {
        std::cerr << "You cannot use [] on a num\n";
        exit(-1);
    }
    if(index >= size[0]) {
        std::cerr << "Index out of range\n";
        exit(-1);
    }
    int other_dim_len = 1;      // Except for the first dimension, the product of other dimensions.
                                // Used to calculate the address after interception
    for(int i = 1; i<(int)size.size(); i++) {
        other_dim_len *= size[i];
    }
    temp.data = data + index*other_dim_len;
    temp.mem_addr = mem_addr;
    temp.cut = 2;
    if(size.size() == 1) {      // If there is only one dimension, it will become a number after interception
        temp.is_num = true;
        temp.size.push_back(1);
    }
    else {
        for (int i = 1; i < (int)size.size(); i++) {
            temp.size.push_back(size[i]);
        }
    }
    // Increase the reference count of temp
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
     * overload operator=
     * Two situations：
     * 1. lvalue is an object interception
     * When assigning A to B[i], we need to check that the size is consistent, and then copy the data
     * in A to B[i](deep copy)
     * 2. lvalue is not an interception
     * When assigning A to B, we need to reduce the reference count of old B, then point B to the memory pointed
     * by A and copy 'size', and then increase the reference count of new B.
     */
    if(this->cut > 0) {         // if lvalue is an interception
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
    else {                      // if lvalue is not an interception
        // reduce the reference count of 'this'
        if(this->mem_addr != nullptr) {
            // Sometimes we create an object without allocate memory for it, and 'mem_addr' is null,
            // we do not need to reduce reference count at this time
            if (counter.find(this->mem_addr) != counter.end()) {
                counter[mem_addr]--;
                if (counter[mem_addr] == 0) {
                    free(mem_addr);
                    counter.erase(mem_addr);
                }
            } else {      // should must found
                std::cerr << "Not found old \'data\' in counter when operator=\n";
                exit(-1);
            }
        }

        // copy attribute of rvalue to lvalue
        this->data = array.data;
        this->mem_addr = array.mem_addr;
        this->size = array.size;
        this->is_num = array.is_num;
        this->cut = 0;

        // increase reference count of new 'this'
        if(counter.find(this->mem_addr) != counter.end()) {
            counter[mem_addr]++;
        }
        else {      // should must found
            std::cerr << "Not found new \'data\' in counter when operator=\n";
            exit(-1);
        }
    }

    return *this;
}

template<typename T>
T Vdarray<T>::to_num() {
    /*
     * if Vdarray is already a num, return its value
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
     * overload =，enables direct assignment to the is_num array
     * For instance, array[0][2][3] = 5;
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
     * set all data in 'data' to 0
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
     * set data in 'data' to random value
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
     * Check old size and new size。
     * Create new object and let its 'data' point to existing 'data', and let its shape be new shape
     */
    if(is_num) {
        std::cerr << "You cannot reshape a num\n";
    }
    // Check new size: negative value no more than 1. No 0. If there is no negative value in new size,
    // the amount of data of new array should be the same as the old one.
    int old_space = 1;
    for(const int &i: size) {
        old_space *= i;
    }
    int negative_count = 0;    // amount of negative value in new_size
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
    // if size check is passed, infer the negative value in new size
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
    // reference count
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
     * print data
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
    // length of new order should equal to length of old size
    if(this->size.size() != new_order.size()) {
        std::cerr << "Axes don't match array\n";
        exit(-1);
    }
    // check new_order: each element should appear only once, and its value should between 0 to n-1
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
            // Do nothing here. Since we have checked the amount of new_order and the value of element in new_order,
            // according to the Drawer Principle, if there is an element did not appear, there must be another
            // element appeared more than once, and was processed in the 'if' block above.
        }
    }
    // create new Vdarray, and let its size be the transposed size
    std::vector<int> new_size;
    for(const int& i: new_order) {
        new_size.push_back(this->size[i]);
    }
    Vdarray<T> temp(new_size);
    // traverse elements in old array(by 'data'), calculate its index, and convert to new index
    int dim = this->size.size();    // dimension
    int old_index[dim];             // old index
    int new_index[dim];             // new index
    int array_len = 1;              // element amount in the array
    for(const int& i: this->size) {
        array_len *= i;
    }
    for(int i = 0; i<dim; i++) {
        old_index[i] = 0;
    }
    for(int index = 0; index<array_len; index++) {  // traverse elements in old array
        // the old index of the element being traversed is 'old_index'
        // map the old index to the new index
        for(int i = 0; i<dim; i++) {
            new_index[i] = old_index[new_order[i]];
        }
        // calculate the position of the element in new array(position: offset from 'data')
        int offset = 0;
        int weight = 1;
        for(int p = dim-1; p>=0; p--) {
            offset += new_index[p] * weight;
            weight *= new_size[p];
        }
        temp.data[offset] = this->data[index];
        // increate old_index by 1
        int p = dim-1;
        old_index[p]++;
        while(true) {
            if(old_index[p] >= this->size[p]) {
                old_index[p] = 0;
                p--;
                if(p < 0) {
                    break;
                }
                old_index[p]++;
            }
            else {
                break;
            }
        }
    }
    return temp;
}

template<typename T>
std::vector<int> Vdarray<T>::shape() {
    /*
     * return size
     */
    return size;
}


#endif //QUANT_NDARRAY_IMPL_H
