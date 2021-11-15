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
 * 每个node对象管理计算图中的一个节点
 * 为了方便Graph使用，我们需要在node中保存output_shape
 *
 * forward:
 * 如Graph中所述，当graph调用Node的forward时，它会传入input和output的指针。Node需要根据自己的数据类型转换指针类型，然后调用
 * 算子的forward
 * 由于在Node中能够确定数据类型，算子类的数据类型应该是确定的
 * 也就是说，以conv2d为例，float32的conv2d和uint8的conv2d应分为两个类。当Node调用算子的forward时，它需要传入input和output的
 * 指针。由于Node的forward接受了output参数，它不需要返回值
 */

class Node {
public:
    int number;                     // 算子编号
    std::string name;               // 算子名称
    void * op;                      // 算子
    std::string dtype;              // 数据类型
    std::vector<int> output_shape;  // output shape

    Node();
    explicit Node(const std::string& read_graph_line,
                  const std::vector<std::vector<int> > &output_shape_list);  // constructor
    ~Node();                                                // destructor

    /*
     * 前向传播函数。Graph在调用node的forward时，将预分配的中间结果空间指针vector
     * 和整个计算图的输入数据指针传入，这些指针类型均为void*，需要forward内部根据算子名称进行处理
     */
    void forward(const std::vector<void*> &intermediate_results, void* input);

    Node* to_qnode();               // 创建量化节点

    void print();                   // 打印节点参数
};

int get_number(const std::string &graph_line);
std::string get_name(const std::string &graph_line);
std::vector<std::string> get_parameters(const std::string &graph_line);

#endif //QUANT_NODE_H
