//
// Created by noname on 2021/10/23.
//

#include "graph.h"
#include "op.h"
#include "tensor.h"

#include <cmath>
#include <cstdio>


Graph::Graph(const std::string& graph_content, const std::string& model_dir)
{
    /*
     * 读取path的文件，根据它创建计算图
     * 每读取一行，创建一个节点
     */
    this->graph_content = graph_content;        // 保存计算图文件内容
    this->model_dir = model_dir;

    // 将计算图内容按\n切分
    std::vector<std::string> graph_lines = split(graph_content, "\n");
    // 遍历计算图的每一行，并创建节点
    for(std::string graph_line : graph_lines) {
        graph_line = delete_annotation(graph_line, "#");
        graph_line = replace(graph_line, " ", "");
        if(graph_line.empty()) {
            continue;
        }
        Node * new_node = new Node(graph_line, output_shape_list, model_dir);  // 不需要delete new_node，因为它后面还会用
        node_list.push_back(new_node);
        output_shape_list.push_back(new_node->output_shape);

        if(new_node->name == OPN_INPUT) {
            this->input_shape = ((Input*)new_node->op)->output_shape;   // 对于input节点，记录input_shape
        }
    }
}

Graph::~Graph() {
    /*
     * destructor: 释放node_list中的所有node
     */
    for(const Node* node: node_list) {
        delete(node);
    }
}

void Graph::fuse_op()
{
    /*
     * 算子融合：将batch_norm2d融合进conv2d中
     * 步骤：考虑到计算图限定了batch_size，而算子融合提供的数据集的图片数量可能与计算图的batch_size不同
     * 所以重新创建一张计算图，新计算图的结构与原始计算图完全相同，但每个算子的尺寸中batch_size改为与输入
     * 数据集的batch_size相同
     * 0. 检查是否包含bn
     * 1. 创建新计算图
     * 2. 使用新计算图和输入数据集进行一次前向传播计算
     * 3. 对bn进行算子融合
     */
    printf("Fusing operators...\n");
    // 0. 检查是否包含bn
    int found = 0;
    for(Node *node: node_list) {
        if(node->name == OPN_NN_BATCH_NORM2D) {
            found = 1;
            break;
        }
    }
    if(found == 0) {
        return;
    }
    // 1. 创建新计算图，暂存原始计算图
    std::string new_graph_content;
    std::vector<std::string> graph_lines = split(graph_content, "\n");
    for(std::string graph_line: graph_lines) {
        // 遍历旧计算图的每一行
        std::string new_graph_line = graph_line;         // 要存入新计算图的算子行
        // 将修改后的行加入new_graph_content
        new_graph_content += new_graph_line;
        new_graph_content.push_back('\n');
    }
    Graph new_graph{new_graph_content, model_dir};
    // 此时前向传播中间结果已经存储在intermediate_results中了
    // 3. 对bn算子进行融合
    /*
     * 遍历this网络，寻找其中的bn层，根据bn层的input_node找到它的输入节点，并检查是否是conv2d
     * 如果检查无误，根据new_graph的conv层输出的中间结果计算running_mean和running_var，再根据融合公式
     * 计算融合后的conv层的权重。此时先不删除刚才的bn层(由于this和new_graph除batch_size之外完全一样，
     * 而要融合的计算图是this，但使用的是new_graph的前向传播数据，为了方便查找数据，需要保证this和new_graph
     * 的各层一一对应，而如果融合了某个bn后立即删除它，则this和new_graph就无法对应了，所以先不删除bn)
     * 所有bn融合完毕后，删除bn层
     */
    for(int layer = 0; layer < (int)this->node_list.size(); layer++) {  // 遍历this网络
        if(this->node_list[layer]->name != OPN_NN_BATCH_NORM2D) {     // 跳过不是bn的层
            continue;
        }
        int bn_input_node = ((Batch_Norm2d*)this->node_list[layer]->op)->input_node; // bn层输入节点编号
        if(this->node_list[bn_input_node]->name != OPN_NN_CONV2D) {
            // 检查bn是否在conv2d之后
            fprintf(stderr, "File graph.cpp, line %d. Only bn after conv2d allowed\n", __LINE__);
            exit(-1);
        }
        // 此时layer指向bn层，bn_input_node指向conv2d层
        // 使用new_graph计算running_mean running_var
        Tensor<float32> running_mean = ((Batch_Norm2d*)this->node_list[layer]->op)->running_mean;
        Tensor<float32> running_var = ((Batch_Norm2d*)this->node_list[layer]->op)->running_var;
        // 提取weight和bias
        Tensor<float32> conv_weight = ((Conv2d*)this->node_list[bn_input_node]->op)->weight.deep_copy();
        Tensor<float32> conv_bias = ((Conv2d*)this->node_list[bn_input_node]->op)->bias.deep_copy();
        Tensor<float32> bn_weight = ((Batch_Norm2d*)this->node_list[layer]->op)->weight.deep_copy();
        Tensor<float32> bn_bias = ((Batch_Norm2d*)this->node_list[layer]->op)->bias.deep_copy();
        // 计算融合后权重
        Tensor<float32> gamma_dot = bn_weight / (running_var + ((Batch_Norm2d*)this->
                node_list[layer]->op)->eps).elewise_sqrt();
        conv_weight = gamma_dot.reshape(std::vector<int>{-1, 1, 1, 1}) * conv_weight;
        conv_bias = gamma_dot * (conv_bias - running_mean) + bn_bias;
        // 将融合后的权重存储进conv
        ((Conv2d*)this->node_list[bn_input_node]->op)->weight = conv_weight;
        ((Conv2d*)this->node_list[bn_input_node]->op)->bias = conv_bias;
    }
    // 现在this的conv层的权重是融合后的权重，new_graph计算的中间结果已经没用了
    // 融合完毕后，删除bn层
    /*
     * 这里很麻烦，因为删除bn之后，各层的number和input_node都需要变。怎么做？
     * 前提：各层的input_node可能不是按顺序排列的，但各层的number一定是按顺序排列的
     * 拿到一个算子，如果是bn则跳过，否则首先修改其number为上一个算子的number+1(即目前已添加的节点数量)
     * 然后修改该节点的输入节点的编号，为：
     *      找到该节点在原始图中的输入节点，如果是bn，则再向上查找一个输入节点。此时找到的这个输入节点是
     *      融合算子后当前节点的真实输入节点。找到这个输入节点之后，计数该输入节点之前删除的节点数量，
     *      用该输入节点编号减去删除的节点数量(即该输入节点算子融合之后的编号)，即为该节点修改后的输入节点编号
     */
    std::vector<Node*> temp_node_list = node_list;  /* 这里只复制了指针，所以在依次对temp_node_list中的每个节点进行处理时
                                                     * 一旦修改了节点的数据，temp_node_list里的数据也会随之改变，但我们需要
                                                     * 节点之间数据传输的原始关系，所以在需要原始关系的时候，从new_graph中读取
                                                     */
    node_list.clear();
    for(Node* i : temp_node_list) {
        // 如果是bn则跳过
        if(i->name == OPN_NN_BATCH_NORM2D) {
            continue;
        }
        // 否则修改其number为上一个算子的number+1
        i->number = (int)node_list.size();
        // 向上查找当前节点的输入节点
        Node* input_node1 = nullptr;
        Node* input_node2 = nullptr;
        if(i->name == OPN_NN_CONV2D) {
            input_node1 = new_graph.node_list[((Conv2d*)i->op)->input_node];
        }
        else if(i->name == OPN_INPUT) {
            // do nothing
        }
        else if(i->name == OPN_NN_RELU) {
            input_node1 = new_graph.node_list[((Relu*)i->op)->input_node];
        }
        else if(i->name == OPN_NN_MAXPOOL2D) {
            input_node1 = new_graph.node_list[((Maxpool2d*)i->op)->input_node];
        }
        else if(i->name == OPN_NN_AVGPOOL2D) {
            input_node1 = new_graph.node_list[((Avgpool2d*)i->op)->input_node];
        }
        else if(i->name == OPN_NN_FLATTEN) {
            input_node1 = new_graph.node_list[((Flatten*)i->op)->input_node];
        }
        else if(i->name == OPN_NN_DENSE) {
            input_node1 = new_graph.node_list[((Dense*)i->op)->input_node];
        }
        else if(i->name == OPN_ADD) {
            input_node1 = new_graph.node_list[((Add*)i->op)->input_node1];
            input_node2 = new_graph.node_list[((Add*)i->op)->input_node2];
        }
        else if(i->name == OPN_CONCAT) {
            input_node1 = new_graph.node_list[((Concat*)i->op)->input_node1];
            input_node2 = new_graph.node_list[((Concat*)i->op)->input_node2];
        }
        else if(i->name == OPN_OUTPUT) {
            input_node1 = new_graph.node_list[((Output*)i->op)->input_node];
        }
        else if(i->name == OPN_NN_DROPOUT) {
            input_node1 = new_graph.node_list[((QDropout*)i->op)->input_node];
        }
        // 如果是bn，则再向上查找一个输入节点
        if(input_node1 != nullptr && input_node1->name == OPN_NN_BATCH_NORM2D) {
            input_node1 = new_graph.node_list[((Batch_Norm2d*)input_node1->op)->input_node];
        }
        if(input_node2 != nullptr && input_node2->name == OPN_NN_BATCH_NORM2D) {
            input_node2 = new_graph.node_list[((Batch_Norm2d*)input_node2->op)->input_node];
        }
        // 计数input_node1和input_node2之前删除的节点数量(即input_node之前的bn数量)
        int count1 = 0;
        if(input_node1 != nullptr) {
            for (int idx = 0; idx < input_node1->number; idx++) {
                if (new_graph.node_list[idx]->name == OPN_NN_BATCH_NORM2D) {
                    count1++;
                }
            }
        }
        int count2 = 0;
        if(input_node2 != nullptr) {
            for(int idx = 0; idx < input_node2->number; idx++) {
                if(new_graph.node_list[idx]->name == OPN_NN_BATCH_NORM2D) {
                    count2++;
                }
            }
        }
        // 用输入节点编号减去删除节点数量
        if(i->name == OPN_NN_CONV2D) {
            ((Conv2d*)i->op)->input_node = input_node1->number - count1;
        }
        else if(i->name == OPN_INPUT) {
            // do nothing
        }
        else if(i->name == OPN_NN_RELU) {
            ((Relu*)i->op)->input_node = input_node1->number - count1;
        }
        else if(i->name == OPN_NN_MAXPOOL2D) {
            ((Maxpool2d*)i->op)->input_node = input_node1->number - count1;
        }
        else if(i->name == OPN_NN_AVGPOOL2D) {
            ((Avgpool2d*)i->op)->input_node = input_node1->number - count1;
        }
        else if(i->name == OPN_NN_FLATTEN) {
            ((Flatten*)i->op)->input_node = input_node1->number - count1;
        }
        else if(i->name == OPN_NN_DENSE) {
            ((Dense*)i->op)->input_node = input_node1->number - count1;
        }
        else if(i->name == OPN_ADD) {
            ((Add*)i->op)->input_node1 = input_node1->number - count1;
            ((Add*)i->op)->input_node2 = input_node2->number - count2;
        }
        else if(i->name == OPN_CONCAT) {
            ((Concat*)i->op)->input_node1 = input_node1->number - count1;
            ((Concat*)i->op)->input_node2 = input_node2->number - count2;
        }
        else if(i->name == OPN_OUTPUT) {
            ((Output*)i->op)->input_node = input_node1->number - count1;
        }
        else if(i->name == OPN_NN_DROPOUT) {
            ((QDropout*)i->op)->input_node = input_node1->number - count1;
        }
        // 将修改后的节点加入节点列表
        node_list.push_back(i);
    }
    printf("Fuse operators finished\n");
}

std::vector<void*> Graph::forward(void *input) {
    /*
     * 前向传播函数：
     * 输入和返回类型实际均为Tensor<>*
     * 该函数应使用intermediate_results存储中间结果，且intermediate_results的实际类型为std::vector<Tensor<>*>
     * 由于Graph不管理类型，所有Tensor<>*均设为void*。调用者必须负责管理类型
     *
     * 计算方式:
     * 1. 不需要初始化intermediate_results。intermediate_results应由调用者管理
     * 2. 遍历graph中所有节点，传入input和output指针
     * 3. 如果某个节点是output节点，那么将它对应的中间结果Tensor数组指针加入一个vector，并最终返回这个vector
     */
    // 使用各节点进行前向传播计算

    
    for(Node *node: node_list) {
        node->forward(intermediate_results, input);
    }
    
    // 将output节点的输出push到ret里
    std::vector<void*> ret;
    for(Node *node: node_list) {
        if(node->name == OPN_OUTPUT) {
            ret.push_back(intermediate_results[node->number]);
        }
        if(node->name == OPN_QOUTPUT) {
            ret.push_back((intermediate_results[node->number]));
        }
    }

    return ret;
}

void Graph::alloc_intermediate_results() {
    /*
     * 为前向传播中间结果分配内存
     */
    for(const Node* node: node_list) {
        if(node->dtype == "float32") {
            Tensor<float32> *inter_res = new Tensor<float32>{node->output_shape};
            inter_res->set_zero();
            intermediate_results.push_back(inter_res);
        }
        else if(node->dtype == "uint8") {
            Tensor<uint8> *inter_res = new Tensor<uint8>{node->output_shape};
            inter_res->set_zero();
            intermediate_results.push_back(inter_res);
        }
    }
}

void Graph::free_intermediate_results() {
    /*
     * 释放前向传播中间结果的内存
     */
    for(int i = 0; i<(int)node_list.size(); i++) {
        if(node_list[i]->dtype == "float32") {
            delete((Tensor<float32>*)intermediate_results[i]);
        }
        if(node_list[i]->dtype == "uint8") {
            delete((Tensor<uint8>*)intermediate_results[i]);
        }
    }
    while(!intermediate_results.empty()) {
        intermediate_results.pop_back();
    }
}

void Graph::print() {
    /*
     * 打印计算图结构
     */
    for(Node * node: node_list) {
        node->print();
    }
}

Graph *Graph::quantization(Tensor<float32>* processed_calib_set) {
    /*
     * 模型量化:
     * 量化过程中需要的数据:
     * rmin, rmax: 原始前向传播中间结果的上下界；通过进行前向传播以统计该数据
     * qmin, qmax: 期望的量化后前向传播中间结果的上下界；手动设定
     * scale: 缩放比例；使用rmin, rmax, qmin, qmax计算
     * zero: 量化后零点；使用qmax, rmax, scale计算
     * rshift: 量化后前向传播计算后右移位数；使用input, weight, output计算
     * coe: 量化后前向传播按原始算法计算后相乘的系数；使用input, weight, output计算
     *
     * 模型量化: 使用scale, zero, weight计算
     * 前向传播: 使用input, weight, 输入、权重、输出的zero、rshift、coe算
     *
     * 只有包含权重数据的层需要量化(conv2d, dense)
     * 需要进行数据放缩的层需要计算rshift, coe(relu, maxpool2d, flatten不需要)
     *
     * 计算方案：
     * 0. 根据当前计算图，创建量化计算图
     * 1. 为各层参数分配空间
     *      对于量化过程中需要的数据rmin, rmax, qmin, qmax, scale, zero, rshift, coe:
     *      其中rmin, ramx, qmin, qmax, scale, zero每层需要一个，有权重的层的权重和偏置数据各额外需要一个
     *      rshift, coe只有部分层需要，且数量不定
     *      量化后前向传播计算时只会用到zero, rshift, coe
     *      所以保存方式设定如下：
     *      对于量化后前向传播计算时需要的参数zero, rshift, coe，在量化算子中保存
     *      对于每层都需要且数量确定的rmin, ramx, qmin, qmax, scale, zero，使用数组保存
     *      注意：这里除了要保存各层中间结果的max min之外，还要保存权重的max min
     *      但权重是不会变的，不需要根据图片计算然后统计。直接单独计算就可以
     * 2. 使用processed_calib_set进行前向传播计算
     *      2.1 计算各层的r, q
     *      2.2 使用r, q计算并累加s, z
     * 3. 求各层平均的s, z
     * 4. 单独计算有权重层的权重的max min参数。不需要使用图片进行前向传播
     * 5. 为需要的算子计算coe, rshift。并将需要的qmin, qmax, scale, zero, coe, rshift存入对应的算子中
     * 6. 对需要的层的weight和bias进行量化
     * 7. 返回量化计算图
     */
    // 0. 根据当前计算图，创建量化计算图
    Graph * qgraph = new Graph();
    for(Node * node: node_list) {
        qgraph->node_list.push_back(node->to_qnode());
    }
    // 1. 为各层参数分配空间(rmax, rmin, qmax, qmin, scale, zero, n, m0)
    int node_number = (int)node_list.size();
    float rmax[node_number];
    float rmin[node_number];
    int qmax[node_number];
    int qmin[node_number];
    float scale[node_number];
    int zero[node_number];
    for(int i = 0; i<node_number; i++) {        // 初始化
        rmax[i] = 0; rmin[i] = 0;
        qmax[i] = 0; qmin[i] = 0;
        scale[i] = 0; zero[i] = 0;
    }
    // 2. 使用processed_calib_set进行前向传播计算
    printf("Calibrating...\n");
    int img_number = processed_calib_set->size[0];
    this->alloc_intermediate_results();
    for(int i = 0; i<img_number; i++) {
        // 2.0 前向传播计算
        Tensor<float32> img = (*processed_calib_set)[i].expand_dim(0);
        this->forward(&img);
        for(int j = 0; j<node_number; j++) {
            // 2.1 计算各层的r, q
            rmax[j] = ((Tensor<float32>*)intermediate_results[j])->max();
            rmin[j] = ((Tensor<float32>*)intermediate_results[j])->min();
            // float temp_rmax = (fabs(rmax[j]) > fabs(rmin[j])) ? fabs(rmax[j]) : fabs(rmin[j]);
            // rmax[j] = temp_rmax;
            // rmin[j] = -temp_rmax;
            qmax[j] = 255;
            qmin[j] = 0;
            float temp_scale = (rmax[j] - rmin[j]) / (float)(qmax[j] - qmin[j]);
            // 2.2 使用r, q计算并累加s, z
            scale[j] += temp_scale;
            zero[j] += (int)std::round((float)qmax[j] - rmax[j]/temp_scale);
        }
        printf("\r%d/%d", i, img_number);
        fflush(stdout);
    }
    printf("\rCalibrate finished\n");
    this->free_intermediate_results();
    // 3. 求各层平均的s, z
    // 注意：部分层应直接使用其输入层的scale和zero
    for(int i = 0; i<node_number; i++) {
        if(node_list[i]->name == OPN_NN_RELU) {
            scale[i] = scale[((Relu*)node_list[i]->op)->input_node];
            zero[i] = zero[((Relu*)node_list[i]->op)->input_node];
        }
        else if(node_list[i]->name == OPN_NN_MAXPOOL2D) {
            scale[i] = scale[((Maxpool2d*)node_list[i]->op)->input_node];
            zero[i] = zero[((Maxpool2d*)node_list[i]->op)->input_node];
        }
        else if(node_list[i]->name == OPN_NN_AVGPOOL2D) {
            scale[i] = scale[((Avgpool2d*)node_list[i]->op)->input_node];
            zero[i] = zero[((Avgpool2d*)node_list[i]->op)->input_node];
        }
        else if(node_list[i]->name == OPN_NN_FLATTEN) {
            scale[i] = scale[((Flatten*)node_list[i]->op)->input_node];
            zero[i] = zero[((Flatten*)node_list[i]->op)->input_node];
        }
        else if(node_list[i]->name == OPN_NN_DROPOUT) {
            // dropout应为输入层scale乘以1-p
            scale[i] = scale[((Dropout*)node_list[i]->op)->input_node] * 
                (1-((Dropout*)node_list[i]->op)->p);
            zero[i] = zero[((Dropout*)node_list[i]->op)->input_node];
        }
        else {
            scale[i] /= (float) img_number;
            zero[i] /= img_number;
        }
    }
    // 4. 计算有权重层的权重的rmin rmax qmin qmax scale zero
    /*
     * 为每一层都分配weight和bias的min max空间。虽然不是每一层都有weight和bias，但这样可以直接通过下标访问
     * 比较方便。用不到的就空着。计算方式为
     * scale_weight = (rmax - rmin) / (qmax - qmin)
     * zero_weight = (int)round(qmax - rmax/scale_weight)
     * scale_bias = scale_weight * scale_input  (本层输入)
     * zero_bias = 0
     */
    float rmax_weight[node_number];
    float rmin_weight[node_number];
    // float rmax_bias[node_number];    // 由于bias的scale是直接根据x和w得到的，因此不需要统计rmax, rmin
    // float rmin_bias[node_number];
    int qmax_weight[node_number];
    int qmin_weight[node_number];
    int qmax_bias[node_number];
    int qmin_bias[node_number];
    float scale_weight[node_number];
    float scale_bias[node_number];
    int zero_weight[node_number];
    int zero_bias[node_number];
    for(int i = 0; i<node_number; i++) {
        rmax_weight[i] = 0; rmin_weight[i] = 0;
        qmax_weight[i] = 0; qmin_weight[i] = 0;
        qmax_bias[i] = 0; qmin_bias[i] = 0;
        scale_weight[i] = 0; scale_bias[i] = 0;
        zero_weight[i] = 0; zero_bias[i] = 0;
    }
    for(int i = 0; i<node_number; i++) {
        if(node_list[i]->name == OPN_NN_CONV2D) {
            rmax_weight[i] = ((Conv2d*)node_list[i]->op)->weight.max();
            rmin_weight[i] = ((Conv2d*)node_list[i]->op)->weight.min();
            float temp_rmax = (fabs(rmax_weight[i]) > fabs(rmin_weight[i])) ? fabs(rmax_weight[i]) : fabs(rmin_weight[i]);
            rmax_weight[i] = temp_rmax;
            rmin_weight[i] = -temp_rmax;
            qmax_weight[i] = 127;
            qmin_weight[i] = -127;
            // qmax_bias[i] = 65535;
            // qmin_bias[i] = -65535;
            qmax_bias[i] = 2147483647;
            qmin_bias[i] = -2147483647;
            scale_weight[i] = (rmax_weight[i] - rmin_weight[i]) / (float)(qmax_weight[i] - qmin_weight[i]);
            zero_weight[i] = (int)std::round((float)qmax_weight[i] - rmax_weight[i]/scale_weight[i]);
            scale_bias[i] = scale_weight[i] * scale[((Conv2d*)node_list[i]->op)->input_node];
            zero_bias[i] = 0;
        }
        else if(node_list[i]->name == OPN_NN_DENSE) {
            rmax_weight[i] = ((Dense*)node_list[i]->op)->weight.max();
            rmin_weight[i] = ((Dense*)node_list[i]->op)->weight.min();
            float temp_rmax = (fabs(rmax_weight[i]) > fabs(rmin_weight[i])) ? fabs(rmax_weight[i]) : fabs(rmin_weight[i]);
            rmax_weight[i] = temp_rmax;
            rmin_weight[i] = -temp_rmax;
            qmax_weight[i] = 127;
            qmin_weight[i] = -127;
            // qmax_bias[i] = 65535;
            // qmin_bias[i] = -65535;
            qmax_bias[i] = 2147483647;
            qmin_bias[i] = -2147483647;
            scale_weight[i] = (rmax_weight[i] - rmin_weight[i]) / (float)(qmax_weight[i] - qmin_weight[i]);
            zero_weight[i] = (int)std::round((float)qmax_weight[i] - rmax_weight[i]/scale_weight[i]);
            scale_bias[i] = scale_weight[i] * scale[((Dense*)node_list[i]->op)->input_node];
            zero_bias[i] = 0;
        }
    }
    // 5. 为需要的算子计算coe, rshift。并将需要的qmin, qmax, scale, zero, coe, rshift存入对应的算子中
    for(int i = 0; i<node_number; i++) {
        if(node_list[i]->name == OPN_NN_CONV2D) {
            calc_m0_n_input_weight(((QConv2d*)qgraph->node_list[i]->op)->coe,
                                   ((QConv2d*)qgraph->node_list[i]->op)->rshift,
                                   scale[((QConv2d*)node_list[i]->op)->input_node],
                                   scale_weight[i], scale[i]);
            ((QConv2d*)qgraph->node_list[i]->op)->zero_x = zero[((QConv2d*)qgraph->node_list[i]->op)->input_node];
            ((QConv2d*)qgraph->node_list[i]->op)->zero_w = zero_weight[i];
            ((QConv2d*)qgraph->node_list[i]->op)->zero_b = zero_bias[i];
            ((QConv2d*)qgraph->node_list[i]->op)->zero_y = zero[i];
            ((QConv2d*)qgraph->node_list[i]->op)->qmin = qmin[i];
            ((QConv2d*)qgraph->node_list[i]->op)->qmax = qmax[i];
        }
        else if(node_list[i]->name == OPN_NN_RELU) {
            ((QRelu*)qgraph->node_list[i]->op)->zero = zero[i];
            ((QRelu*)qgraph->node_list[i]->op)->qmax = qmax[i];
        }
        else if(node_list[i]->name == OPN_NN_MAXPOOL2D) {
            ((QMaxpool2d*)qgraph->node_list[i]->op)->zero = zero[i];
        }
        else if(node_list[i]->name == OPN_NN_AVGPOOL2D) {
            ((QAvgpool2d*)qgraph->node_list[i]->op)->zero = zero[i];
        }
        else if(node_list[i]->name == OPN_NN_DROPOUT) {
            ((QDropout*)qgraph->node_list[i]->op)->zero = zero[i];
        }
        else if(node_list[i]->name == OPN_NN_DENSE) {
            calc_m0_n_input_weight(((QDense*)qgraph->node_list[i]->op)->coe,
                                   ((QDense*)qgraph->node_list[i]->op)->rshift,
                                   scale[((QDense*)node_list[i]->op)->input_node],
                                   scale_weight[i], scale[i]);
            ((QDense*)qgraph->node_list[i]->op)->zero_x = zero[((QDense*)qgraph->node_list[i]->op)->input_node];
            ((QDense*)qgraph->node_list[i]->op)->zero_w = zero_weight[i];
            ((QDense*)qgraph->node_list[i]->op)->zero_b = zero_bias[i];
            ((QDense*)qgraph->node_list[i]->op)->zero_y = zero[i];
            ((QDense*)qgraph->node_list[i]->op)->qmin = qmin[i];
            ((QDense*)qgraph->node_list[i]->op)->qmax = qmax[i];
        }
        else if(node_list[i]->name == OPN_ADD) {
            calc_m0_n_input_input(((QAdd*)qgraph->node_list[i]->op)->coe1,
                                  ((QAdd*)qgraph->node_list[i]->op)->coe2,
                                  ((QAdd*)qgraph->node_list[i]->op)->rshift1,
                                  ((QAdd*)qgraph->node_list[i]->op)->rshift2,
                                  scale[((QAdd*)node_list[i]->op)->input_node1],
                                  scale[((QAdd*)node_list[i]->op)->input_node2],
                                  scale[i]);
            ((QAdd*)qgraph->node_list[i]->op)->zero_x1 = zero[((QAdd*)qgraph->node_list[i]->op)->input_node1];
            ((QAdd*)qgraph->node_list[i]->op)->zero_x2 = zero[((QAdd*)qgraph->node_list[i]->op)->input_node2];
            ((QAdd*)qgraph->node_list[i]->op)->zero_y = zero[i];
            ((QAdd*)qgraph->node_list[i]->op)->qmin = qmin[i];
            ((QAdd*)qgraph->node_list[i]->op)->qmax = qmax[i];
        }
        else if(node_list[i]->name == OPN_CONCAT) {
            calc_m0_n_input_input(((QConcat*)qgraph->node_list[i]->op)->coe1,
                                  ((QConcat*)qgraph->node_list[i]->op)->coe2,
                                  ((QConcat*)qgraph->node_list[i]->op)->rshift1,
                                  ((QConcat*)qgraph->node_list[i]->op)->rshift2,
                                  scale[((QConcat*)node_list[i]->op)->input_node1],
                                  scale[((QConcat*)node_list[i]->op)->input_node2],
                                  scale[i]);
            ((QConcat*)qgraph->node_list[i]->op)->zero_x1 = zero[((QConcat*)qgraph->node_list[i]->op)->input_node1];
            ((QConcat*)qgraph->node_list[i]->op)->zero_x2 = zero[((QConcat*)qgraph->node_list[i]->op)->input_node2];
            ((QConcat*)qgraph->node_list[i]->op)->zero_y = zero[i];
            ((QConcat*)qgraph->node_list[i]->op)->qmin = qmin[i];
            ((QConcat*)qgraph->node_list[i]->op)->qmax = qmax[i];
        }
    }
    // 6. 对需要的层的weight和bias进行量化
    for(int i = 0; i<node_number; i++) {
        if(node_list[i]->name == OPN_NN_CONV2D) {
            quant(((QConv2d*)qgraph->node_list[i]->op)->weight, ((Conv2d*)node_list[i]->op)->weight,
                  scale_weight[i], zero_weight[i], qmin_weight[i], qmax_weight[i]);
            quant(((QConv2d*)qgraph->node_list[i]->op)->bias, ((Conv2d*)node_list[i]->op)->bias,
                  scale_bias[i], zero_bias[i], qmin_bias[i], qmax_bias[i]);
        }
        else if(node_list[i]->name == OPN_NN_DENSE) {
            quant(((QDense*)qgraph->node_list[i]->op)->weight, ((Dense*)node_list[i]->op)->weight,
                  scale_weight[i], zero_weight[i], qmin_weight[i], qmax_weight[i]);
            quant(((QDense*)qgraph->node_list[i]->op)->bias, ((Dense*)node_list[i]->op)->bias,
                  scale_bias[i], zero_bias[i], qmin_bias[i], qmax_bias[i]);
        }
    }
    // 7. 返回量化计算图
    return qgraph;
}

Graph::Graph() {
    /*
     * 创建空的计算图
     */
    // donothing
}

void Graph::save(std::string path) {
    /*
     * 保存计算图
     */
    // 检查输出文件夹是否存在，不存在就创建一个
    if(access(path.c_str(), 0) < 0) {
        std::string cmd = "mkdir " + path;
        int ret = system(cmd.c_str());
        if(ret != 0) {
            fprintf(stderr, "file graph.cpp line %d: Error return value %d\n", __LINE__, ret);
        }
    }
    // 创建或清空计算图文件
    if(path[path.size()-1] != '/') {
        path += "/";
    }
    FILE * file = fopen((path+GRAPH_FILE_NAME).c_str(), "w");
    fclose(file);
    // 存储计算图
    for(Node * node: node_list) {
        node->save(path);
    }
}