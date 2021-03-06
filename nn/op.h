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
 * Maxpool2d: nn.maxpool2d
 * Avgpool2d: nn.avgpool2d
 * Input: input
 * Flatten: nn.flatten
 * Dense: nn.dense
 * Output: output
 * Add: add
 * Concat: concat
 * Batch_Normalization: nn.batch_norm
 *
 * QConv2d: nn.qconv2d
 * QRelu: nn.qrelu
 * QMaxpool2d: nn.qmaxpool2d
 * QAvgpool2d: nn.qavgpool2d
 * QInput: qinput
 * QFlatten: qflatten
 * QDense: nn.qdense
 * QOutput: qoutput
 * QAdd: qadd
 * QConcat: qconcat
 *
 * 如果要添加新的算子，你需要修改：
 * 0. Node name #define list
 * 1. Node::Node()
 * 2. Node::~Node()
 * 3. Node::forward()
 * 4. Node::print()
 * 5. Node::to_qnode()
 * 6. Node::save()
 * 7. node.cpp  get_name()
 * 8. Graph::fuse_op()      2处
 * 9. Graph::forward()
 * 10. Graph::quantization()     5处
 */

#define OPN_INPUT                   1
#define OPN_NN_CONV2D               2
#define OPN_NN_MAXPOOL2D            3
#define OPN_NN_RELU                 4
#define OPN_NN_FLATTEN              5
#define OPN_NN_DENSE                6
#define OPN_ADD                     7
#define OPN_CONCAT                  8
#define OPN_OUTPUT                  9
#define OPN_NN_BATCH_NORM2D         10
#define OPN_NN_AVGPOOL2D            11
#define OPN_NN_DROPOUT              12

#define OPN_QINPUT                  1001
#define OPN_NN_QCONV2D              1002
#define OPN_NN_QMAXPOOL2D           1003
#define OPN_NN_QRELU                1004
#define OPN_NN_QFLATTEN             1005
#define OPN_NN_QDENSE               1006
#define OPN_QADD                    1007
#define OPN_QCONCAT                 1008
#define OPN_QOUTPUT                 1009
#define OPN_NN_QAVGPOOL2D           1010
#define OPN_NN_QDROPOUT             1011

#define GRAPH_FILE_NAME             "graph.txt"



#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <cassert>

#include "util.h"
#include "tensor.h"
#include "functional.h"
#include "fixed_point.h"

namespace F = functional;

class Input {
public:
    std::vector<int> output_shape;      // 输出尺寸
    explicit Input(const std::vector<std::string>& parameters);             // constructor
    ~Input();
    void forward(Tensor<float32> *input, Tensor<float32> *output);
    void print();
    void save(const std::string &path, int number);
};

class QInput {
public:
    std::vector<int> output_shape;
    explicit QInput(Input * op);
    ~QInput();
    void forward(Tensor<uint8> *input, Tensor<uint8> *output);
    void print();
    void save(const std::string &path, int number);
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
           const std::vector<std::vector<int> > &output_shape_list,
           const std::string& model_dir);       // constructor
    ~Conv2d();
    void forward(Tensor<float32> *input, Tensor<float32> *output);
    void print();
    void save(const std::string &path, int number);
};

class QConv2d {
public:
    int input_node;
    Tensor<int8> weight;
    Tensor<int32> bias;
    int output_channel;
    int input_channel;
    std::vector<int> kernel_size;
    std::vector<int> stride;
    std::vector<int> padding;
    std::vector<int> dilation;
    std::vector<int> output_shape;
    int zero_x;
    int zero_w;
    int zero_b;
    int zero_y;
    Fixed_point coe;
    int rshift;
    int qmin;
    int qmax;
    explicit QConv2d(Conv2d* op);
    ~QConv2d();
    void forward(Tensor<uint8> *input, Tensor<uint8> *output);
    void print();
    void save(const std::string &path, int number);
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
    void print();
    void save(const std::string &path, int number);
};

class QMaxpool2d {
public:
    int input_node;
    std::vector<int> kernel_size;
    std::vector<int> stride;
    std::vector<int> padding;
    std::vector<int> dilation;
    std::vector<int> output_shape;
    int zero;
    explicit QMaxpool2d(Maxpool2d * op);
    ~QMaxpool2d();
    void forward(Tensor<uint8> *input, Tensor<uint8> *output);
    void print();
    void save(const std::string &path, int number);
};

class Relu {
public:
    int input_node;                     // 输入节点编号
    std::vector<int> output_shape;      // 输出尺寸
    Relu(const std::vector<std::string>& parameters,
         const std::vector<std::vector<int> > &output_shape_list);     // constructor
    ~Relu();
    void forward(Tensor<float32> *input, Tensor<float32> *output);
    void print();
    void save(const std::string &path, int number);
};

class QRelu {
public:
    int input_node;
    std::vector<int> output_shape;
    int zero;
    int qmax;
    explicit QRelu(Relu * op);
    ~QRelu();
    void forward(Tensor<uint8> *input, Tensor<uint8> *output);
    void print();
    void save(const std::string &path, int number);
};

class Flatten {
public:
    int input_node;                     // 输入节点编号
    std::vector<int> output_shape;      // 输出尺寸
    Flatten(const std::vector<std::string> &parameters,
            const std::vector<std::vector<int> > &output_shape_list);  // constructor
    ~Flatten();
    void forward(Tensor<float32> *input, Tensor<float32> *output);
    void print();
    void save(const std::string &path, int number);
};

class QFlatten {
public:
    int input_node;
    std::vector<int> output_shape;
    explicit QFlatten(Flatten * op);
    ~QFlatten();
    void forward(Tensor<uint8> *input, Tensor<uint8> *output);
    void print();
    void save(const std::string &path, int number);
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
          const std::vector<std::vector<int> > &output_shape_list,
          const std::string& model_dir);     // constructor
    ~Dense();
    void forward(Tensor<float32> *input, Tensor<float32> *output);
    void print();
    void save(const std::string &path, int number);
};

class QDense {
public:
    int input_node;
    Tensor<int8> weight;
    Tensor<int32> bias;
    int output_channel;
    int input_channel;
    std::vector<int> output_shape;
    int zero_x;
    int zero_w;
    int zero_b;
    int zero_y;
    Fixed_point coe;
    int rshift;
    int qmin;
    int qmax;
    explicit QDense(Dense *op);
    ~QDense();
    void forward(Tensor<uint8> *input, Tensor<uint8> *output);
    void print();
    void save(const std::string &path, int number);
};

class Dropout {
public:
    int input_node;                         // 输入节点编号
    float p;                                // 丢弃比例
    std::vector<int> output_shape;          // 输出尺寸
    Dropout(const std::vector<std::string> &parameters,
            const std::vector<std::vector<int> > &output_shape_list);       // constructor
    ~Dropout();
    void forward(Tensor<float32> *input, Tensor<float32> *output);
    void print();
    void save(const std::string &path, int number);
};

class QDropout {
public:
    int input_node;
    float p;
    int zero;
    std::vector<int> output_shape;
    QDropout(Dropout * op);
    ~QDropout();
    void forward(Tensor<uint8> *input, Tensor<uint8> *output);
    void print();
    void save(const std::string &path, int number);
};

class Output {
public:
    int input_node;                     // 输入节点编号
    std::vector<int> output_shape;      // 输出尺寸
    Output(const std::vector<std::string> &parameters,
           const std::vector<std::vector<int> > &output_shape_list);        // constructor
    ~Output();
    void forward(Tensor<float32> *input, Tensor<float32> *output);
    void print();
    void save(const std::string &path, int number);
};

class QOutput {
public:
    int input_node;
    std::vector<int> output_shape;
    explicit QOutput(Output * op);
    ~QOutput();
    void forward(Tensor<uint8> *input, Tensor<uint8> *output);
    void print();
    void save(const std::string &path, int number);
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
    void print();
    void save(const std::string &path, int number);
};

class QAdd {
public:
    int input_node1;
    int input_node2;
    std::vector<int> output_shape;
    int zero_x1;
    int zero_x2;
    int zero_y;
    Fixed_point coe1;
    Fixed_point coe2;
    int rshift1;
    int rshift2;
    int qmin;
    int qmax;
    explicit QAdd(Add * op);
    ~QAdd();
    void forward(Tensor<uint8> *input1, Tensor<uint8> *input2, Tensor<uint8> *output);
    void print();
    void save(const std::string &path, int number);
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
    void print();
    void save(const std::string &path, int number);
};

class QConcat {
public:
    int input_node1;
    int input_node2;
    int dim;
    std::vector<int> output_shape;
    int zero_x1;
    int zero_x2;
    int zero_y;
    Fixed_point coe1;
    Fixed_point coe2;
    int rshift1;
    int rshift2;
    int qmin;
    int qmax;
    explicit QConcat(Concat * op);
    ~QConcat();
    void forward(Tensor<uint8> *input1, Tensor<uint8> *input2, Tensor<uint8> *output);
    void print();
    void save(const std::string &path, int number);
};

class Batch_Norm2d {
public:
    int input_node;
    int num_features;
    float eps=1e-5;
    float momentum=0.1;
    std::string weight_path;
    Tensor<float32> weight;
    std::string bias_path;
    Tensor<float32> bias;
    std::string running_mean_path;
    Tensor<float32> running_mean;
    std::string running_var_path;
    Tensor<float32> running_var;
    std::vector<int> output_shape;
    Batch_Norm2d(const std::vector<std::string> &parameters,
                 const std::vector<std::vector<int> > &output_shape_list,
                 const std::string& model_dir);       // constructor
    ~Batch_Norm2d();
    void forward(Tensor<float32> *input, Tensor<float32> *output);
    void print();
    void save(const std::string &path, int number);
};

class Avgpool2d{
public:
    int input_node;                     // 输入节点编号
    std::vector<int> kernel_size;
    std::vector<int> stride;
    std::vector<int> padding;
    std::vector<int> output_shape;
    Avgpool2d(const std::vector<std::string> &parameters,
              const std::vector<std::vector<int> > &output_shape_list);    // constructor
    ~Avgpool2d();
    void forward(Tensor<float32> *input, Tensor<float32> *output);
    void print();
    void save(const std::string &path, int number);
};

class QAvgpool2d {
public:
    int input_node;
    std::vector<int> kernel_size;
    std::vector<int> stride;
    std::vector<int> padding;
    std::vector<int> output_shape;
    int zero;
    explicit QAvgpool2d(Avgpool2d * op);
    ~QAvgpool2d();
    void forward(Tensor<uint8> *input, Tensor<uint8> *output);
    void print();
    void save(const std::string &path, int number);
};

#endif //QUANT_OP_H
