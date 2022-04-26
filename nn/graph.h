//
// Created by noname on 2021/10/23.
//

#ifndef QUANT_GRAPH_H
#define QUANT_GRAPH_H


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <unistd.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/hal/interface.h>

#include "op.h"
#include "tensor.h"
#include "node.h"
#include "util.h"
#include "preprocess.h"
#include "fixed_point.h"
#include "quant_tools.h"

/*
 * 计算图：
 * 计算图使用vector存储节点列表，里面每个元素是一个节点
 * 当调用forward进行推理时，我们将预先存储好的Tensor类型的input数据的指针传入forward方法，经过计算后返回Tensor数据
 * Graph不管理类型，所以无法知道Tensor数据类型，因此传输指针时使用Void*。需要在内部真正使用数据时根据情况改变指针类型
 *
 * forward的计算思路：
 * 考虑到需要处理大量图片，如果每张图片计算过程中都需要重新申请中间结果存储空间，必然十分浪费时间
 * 所以我们需要在计算之前提前申请好存储空间，在每张图片计算过程中重复利用这些空间。那么就必须能够在Graph中就知道数据类型(flot32? uint8)
 * 和每个节点的output尺寸，那么我们最好在Node中保存这两个数据。这样在forward计算的时候可以按照以下步骤进行：
 * 1. 根据Node的dtype和output尺寸为每个Node的输出申请空间(Tensor类型)
 * 2. 遍历每一个Node，将输入(上一层的输出空间指针)和输出空间指针传入(Node内部自动根据算子类型进行计算)
 * 3. 得到最后结果后，将结果指针返回(void*)
 * 调用forward的函数自行根据上下文处理返回值类型
 */



class Graph {
public:
    std::string graph_content;          // 计算图文件内容
    std::string model_dir;              // 模型文件夹路径
    std::vector<Node*> node_list;        // 节点列表
    std::vector<int> input_shape;       // 输入尺寸

    /*
     * 后面的节点在推断output_shape时需要用到前面节点的output shape，所以需要传入前面节点的output_shape。本想直接传入
     * node_list，但这会导致node和op相互include，所以新建一个vector用于存储各节点的output_shape，然后传入这个vector
     */
    std::vector<std::vector<int> > output_shape_list;

    /*
     * 这个vector用于存储forward过程中的中间结果，即各节点的输出
     * vector里存储的实际类型是Tensor<>*，但由于graph不处理类型，所以将它设为void*。forward函数需要自己处理类型
     */
    std::vector<void*> intermediate_results;

    explicit Graph();
    Graph(const std::string& path, const std::string& model_dir);                // constructor
    ~Graph();

    /*
     * 由于forward()每次只能计算一个batch，如果在forward()里为中间结果分配空间，那么每次调用forward都要重新分配空间，会
     * 浪费大量时间。所以提供此方法，可在调用forward的函数中分配空间
     * 同时提供释放空间的方法
     */
    void alloc_intermediate_results();
    void free_intermediate_results();

    /*
     * 前向传播函数。由于graph不限制数据类型(float32 uint8等)，这里只返回std::vector<void*>。实际返回类型为
     * std::vector<Tensor<>*>。调用者需要根据上下文修改指针类型
     * 考虑到某些神经网络可能有超过1个输出节点，这里使用vector存储返回值
     */
     std::vector<void*> forward(void * input);

    /*
     * 融合算子。将batch_norm2d融入conv2d
     * 只能在dtype=="float32"时使用此函数
     */
     void fuse_op();

     Graph* quantization(Tensor<float32>* processed_calib_set);

     void print();              // 打印计算图结构

     void save(std::string path);               // 保存计算图

};




#endif //QUANT_GRAPH_H
