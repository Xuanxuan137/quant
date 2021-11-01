//
// Created by noname on 2021/10/28.
//

#ifndef QUANT_NODE_H
#define QUANT_NODE_H

#include <iostream>
#include <cstdio>
#include <string>
#include <cassert>

#include "util.h"
#include "op.h"

/*
 * Node:
 * Each node object manage a node in the calculation graph.
 * In order to facilitate Graph to use it, we need to store dtype and output_shape data in Node
 *
 * forward:
 * As described by Graph, when graph calls Node's forward, it will pass in input data and output data pointers.
 * Node should convert the pointer type according to the data type of the node, and then call forward of the
 * operator for forward propagation calculation
 * Since the data type can be determined in Node, the type of the operator class should be determined.
 * That is, taking conv2d as an example, conv2d of float32 and conv2d of uint8 should be divided into two classes
 * When Node calls the forward of the operator, it needs to pass in the input and output pointers
 * Since Node's forward accepts 'output' parameters, there is no need to return a value
 */

class Node {
public:
    int number;                     // operator serial number
    std::string name;               // operator name
    void * op;                      // operator object pointer
    std::string dtype;              // data type
    std::vector<int> output_shape;  // output shape
    explicit Node(const std::string& read_graph_line,
                  const std::vector<std::vector<int> > &output_shape_list);  // constructor
    ~Node();                                                // destructor
//    void forward(void * input, void* output);             // Forward propagation function. Since the data type
//                                                             is not fixed, only void* is passed in and out,
//                                                             Need to automatically determine the data type internally
};

int get_number(const std::string &graph_line);
std::string get_name(const std::string &graph_line);
std::vector<std::string> get_parameters(const std::string &graph_line);

#endif //QUANT_NODE_H
