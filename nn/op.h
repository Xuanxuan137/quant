//
// Created by noname on 2021/10/29.
//

#ifndef QUANT_OP_H
#define QUANT_OP_H

/*
 * 每个类是一个具体的算子，不同算子使用不同的类
 *
 * forward():
 * 每个算子应有一个forward方法，该方法接受两个参数：input 和 output，其返回值类型为Tensor<>*
 * forward使用input和其他属性进行计算，并将结果存入output(已分配好空间)中
 * 不需要返回值
 */

/*
 * 支持的算子: 算子名称
 * Conv2d: nn.conv2d
 * Relu: nn.relu
 * Input: input
 * Flatten: nn.flatten
 * Dense: nn.dense
 * Output: output
 * Add: add
 * Concat: concat
 *
 * QConv2d
 * QRelu
 * QInput
 * QFlatten
 * QDense
 * QOutput
 * QAdd
 * QConcat
 *
 * 如果要添加新的算子，你需要修改：
 * 1. Node::Node()
 * 2. Node::~Node()
 * 3. Graph::forward()
 */


#define OPN_INPUT               1
#define OPN_NN_CONV2D           2
#define OPN_NN_RELU             3
#define OPN_NN_FLATTEN          4
#define OPN_NN_DENSE            5
#define OPN_OUTPUT              6
#define OPN_ADD                 7
#define OPN_CONCAT              8
#define OPN_MAXPOOL2D           9

#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <cassert>

#include "util.h"
#include "tensor.h"
#include "functional.h"

namespace F = functional;

class Input {
public:
    std::vector<int> output_shape;      // 输出尺寸
    explicit Input(const std::vector<std::string>& parameters);             // constructor
    ~Input();
    void forward(Tensor<float32> *input, Tensor<float32> *output);
};

class Conv2d {
public:
    int input_node;                     // 输入节点编号
    std::string weight_path;            // weight file path
    Tensor<float32> weight;            // weight
    std::string bias_path;              // bias file path
    Tensor<float32> bias;              // bias
    int output_channel;
    int input_channel;
    std::vector<int> kernel_size;       // kernel_size
    std::vector<int> stride;
    std::vector<int> padding;
    std::vector<int> dilation;
    std::vector<int> output_shape;      // 输出尺寸
    Conv2d(const std::vector<std::string>& parameters,
           const std::vector<std::vector<int> > &output_shape_list);       // constructor
    ~Conv2d();
    void forward(Tensor<float32> *input, Tensor<float32> *output);
};

class Maxpool2d {
public:
    int input_node;                     // 输入节点编号
    std::vector<int> kernel_size;
    std::vector<int> stride;
    std::vector<int> padding;
    std::vector<int> dilation;
    std::vector<int> output_shape;
    Maxpool2d(const std::vector<std::string> &parameters,
              const std::vector<std::vector<int> > &output_shape_list);    // constructor
    ~Maxpool2d();
    void forward(Tensor<float32> *input, Tensor<float32> *output);
};

class Relu {
public:
    int input_node;                     // 输入节点编号
    std::vector<int> output_shape;      // 输出尺寸
    Relu(const std::vector<std::string>& parameters,
         const std::vector<std::vector<int> > &output_shape_list);     // constructor
    ~Relu();
    void forward(Tensor<float32> *input, Tensor<float32> *output);
};


class Flatten {
public:
    int input_node;                     // 输入节点编号
    std::vector<int> output_shape;      // 输出尺寸
    Flatten(const std::vector<std::string> &parameters,
            const std::vector<std::vector<int> > &output_shape_list);  // constructor
    ~Flatten();
    void forward(Tensor<float32> *input, Tensor<float32> *output);
};


class Dense {
public:
    int input_node;                     // 输入节点编号
    std::string weight_path;
    Tensor<float32> weight;
    std::string bias_path;
    Tensor<float32> bias;
    int output_channel;
    int input_channel;
    std::vector<int> output_shape;      // 输出尺寸
    Dense(const std::vector<std::string> &parameters,
          const std::vector<std::vector<int> > &output_shape_list);     // constructor
    ~Dense();
    void forward(Tensor<float32> *input, Tensor<float32> *output);
};


class Output {
public:
    int input_node;                     // 输入节点编号
    std::vector<int> output_shape;      // 输出尺寸
    Output(const std::vector<std::string> &parameters,
           const std::vector<std::vector<int> > &output_shape_list);        // constructor
    ~Output();
    void forward(Tensor<float32> *input, Tensor<float32> *output);
};


class Add {
public:
    int input_node1;
    int input_node2;
    std::vector<int> output_shape;
    Add(const std::vector<std::string> &parameters,
        const std::vector<std::vector<int> > &output_shape_list);           // constructor
    ~Add();
    void forward(Tensor<float32> *input1, Tensor<float32> *input2, Tensor<float32> *output);
};


class Concat {
public:
    int input_node1;
    int input_node2;
    int dim;
    std::vector<int> output_shape;
    Concat(const std::vector<std::string> &parameters,
           const std::vector<std::vector<int> > &output_shape_list);        // constructor
    ~Concat();
    void forward(Tensor<float32> *input1, Tensor<float32> *input2, Tensor<float32> *output);
};

#endif //QUANT_OP_H
