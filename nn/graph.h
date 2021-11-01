//
// Created by noname on 2021/10/23.
//

#ifndef QUANT_GRAPH_H
#define QUANT_GRAPH_H

#include <cstdio>
#include <iostream>
#include <string>
#include <fstream>

#include "vdarray.h"
#include "node.h"
#include "util.h"

/*
 * 计算图：
 * 计算图使用vector存储节点列表，里面每个元素是一个节点
 * 当调用forward进行推理时，我们将预先存储好的Vdarray类型的input数据的指针传入forward方法，经过计算后返回Vdarray数据
 * Graph不管理类型，所以无法知道Vdarray数据类型，因此传输指针时使用Void*。需要在内部真正使用数据时根据情况改变指针类型
 *
 * forward的计算思路：
 * 考虑到需要处理大量图片，如果每张图片计算过程中都需要重新申请中间结果存储空间，必然十分浪费时间
 * 所以我们需要在计算之前提前申请好存储空间，在每张图片计算过程中重复利用这些空间。那么就必须能够在Graph中就知道数据类型(flot32? uint8)
 * 和每个节点的output尺寸，那么我们最好在Node中保存这两个数据。这样在forward计算的时候可以按照以下步骤进行：
 * 1. 根据Node的dtype和output尺寸为每个Node的输出申请空间(Vdarray类型)
 * 2. 遍历每一个Node，将输入(上一层的输出空间指针)和输出空间指针传入(Node内部自动根据算子类型进行计算)
 * 3. 得到最后结果后，将结果指针返回(void*)
 * 调用forward的函数自行根据上下文处理返回值类型
 */

/*
 * Calculation Graph:
 * Calculation graph use vector to store node lists, in which each element is a node
 * When call forward to inference, we transfer the pre-stored input data which is in Vdarray type into forward(),
 * and return Vdarray data after calculate.
 * Graph does not manage data type, so it does not know the data type of Vdarray, so we use void* to transfer
 * pointer. We need to change the type of the pointer when use the data.
 *
 * The calculation idea of forward:
 * Taking into account the need to process a large number of pictures, if each picture needs to be re-applied
 * for storage space for intermediate results during the calculation process, it will inevitably be a waste of time
 * Therefore, we need to apply for storage space in advance before calculation, and reuse this space during the
 * calculation of each picture. Then we must be able to know the data type in Graph (float32? uint8?) and the
 * output size of each node, then we'd better save these two data in Node. In this way, the following
 * steps can be followed during forward calculation:
 * 1. Apply for space (Vdarray type) for each Node's output according to Node's dtype and output size
 * 2. Traverse each Node and pass in the input (the output space pointer of the previous layer) and the output
 * space pointer (the Node will automatically calculate according to the operator type)
 * 3. After getting the final result, return the result pointer (void*)
 * The function that calls forward handles the return value type according to the context by itself
 */


class Graph {
public:
    std::vector<Node> node_list;        // node list
    std::vector<std::vector<int> > output_shape_list;
    /*
     * Later nodes need to use the output size of the previous node in the analysis, so the output size
     * of the previous node needs to be passed to the constructor of Node. Originally I wanted to directly
     * pass the node_list in, but because it needs to be passed to the op, it is necessary to include op
     * and node each other, which is very troublesome to deal with, so create a new vector and save the
     * output_shape of each node
     */
    explicit Graph(const std::string& path);                // constructor
    // void * forward(void * input);
    /*
     * Forward propagation function. Since graph does not limit the data type (float32 or uint8, etc.),
     * only void* is returned here. The actual return is Vdarray<>*. Need to modify the pointer type
     * according to the situation
     */
};


#endif //QUANT_GRAPH_H
