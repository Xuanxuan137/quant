//
// Created by noname on 2021/10/29.
//

#ifndef QUANT_OP_H
#define QUANT_OP_H

/*
 * Each class is a specific operator, and different classes are designed for the same operators of
 * different data types
 *
 * forward():
 * Each class should have a forward method that accepts input and output parameters, and its type
 * should be of type Vdarray<>*
 * forward uses input and other attributes of the operator to calculate, and stores the result in output
 * No return value required
 */

/*
 * Supported operators:
 * Conv2d
 * Relu
 * Input
 * Flatten
 * Dense
 * Output
 * Add
 * Concat
 *
 * QConv2d
 * QRelu
 * QInput
 * QFlatten
 * QDense
 * QOutput
 * QAdd
 * QConcat
 */

#include <iostream>
#include <vector>
#include <string>
#include <cstdio>

#include "util.h"
#include "node.h"
#include "vdarray.h"

class Input {
public:
    std::vector<int> output_shape;      // output shape
    explicit Input(const std::vector<std::string>& parameters);             // constructor
//    ~Input();
};

class Conv2d {
public:
    int input_node;                     // input node serial number
    std::string weight_path;            // weight file path
    Vdarray<float> weight;              // weight
    std::string bias_path;              // bias file path
    Vdarray<float> bias;                // bias
    int output_channel;
    int input_channel;
    std::vector<int> kernel_size;       // kernel_size
    std::vector<int> stride;
    std::vector<int> padding;
    std::vector<int> dilation;
    std::vector<int> output_shape;      // output shape
    Conv2d(const std::vector<std::string>& parameters,
           const std::vector<std::vector<int> > &output_shape_list);       // constructor
};

class Maxpool2d {
public:
    int input_node;                     // input node serial number
    std::vector<int> kernel_size;
    std::vector<int> stride;
    std::vector<int> padding;
    std::vector<int> dilation;
    std::vector<int> output_shape;
    Maxpool2d(const std::vector<std::string> &parameters,
              const std::vector<std::vector<int> > &output_shape_list);    // constructor
};

class Relu {
public:
    int input_node;                     // input node serial number
    std::vector<int> output_shape;      // output shape
    Relu(const std::vector<std::string>& parameters,
         const std::vector<std::vector<int> > &output_shape_list);     // constructor
};


class Flatten {
public:
    int input_node;                     // input node serial number
    std::vector<int> output_shape;      // output shape
    Flatten(const std::vector<std::string> &parameters,
            const std::vector<std::vector<int> > &output_shape_list);  // constructor
};


class Dense {
public:
    int input_node;                     // input node serial number
    std::string weight_path;
    Vdarray<float> weight;
    std::string bias_path;
    Vdarray<float> bias;
    int output_channel;
    int input_channel;
    std::vector<int> output_shape;      // output shape
    Dense(const std::vector<std::string> &parameters,
          const std::vector<std::vector<int> > &output_shape_list);     // constructor
};


class Output {
public:
    int input_node;                     // input node serial number
    std::vector<int> output_shape;      // output shape
    Output(const std::vector<std::string> &parameters,
           const std::vector<std::vector<int> > &output_shape_list);        // constructor
};


class Add {
public:
    int input_node1;
    int input_node2;
    std::vector<int> output_shape;
    Add(const std::vector<std::string> &parameters,
        const std::vector<std::vector<int> > &output_shape_list);           // constructor
};


class Concat {
public:
    int input_node1;
    int input_node2;
    int dim;
    std::vector<int> output_shape;
    Concat(const std::vector<std::string> &parameters,
           const std::vector<std::vector<int> > &output_shape_list);        // constructor
};

#endif //QUANT_OP_H
