//
// Created by noname on 2021/10/23.
//

#include "graph.h"


Graph::Graph(const std::string& graph_content)
{
    /*
     * 读取path的文件，根据它创建计算图
     * 每读取一行，创建一个节点
     */
    this->graph_content = graph_content;        // 保存计算图文件内容

    // 将计算图内容按\n切分
    std::vector<std::string> graph_lines = split(graph_content, "\n");
    // 遍历计算图的每一行，并创建节点
    for(std::string graph_line : graph_lines) {
        graph_line = delete_annotation(graph_line, "#");
        graph_line = replace(graph_line, " ", "");
        if(graph_line.empty()) {
            continue;
        }
        Node * new_node = new Node(graph_line, output_shape_list);  // 不需要delete new_node，因为它后面还会用
        node_list.push_back(new_node);
        output_shape_list.push_back(new_node->output_shape);
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

void Graph::fuse_op(Tensor<float32> *calc_running_img)
{
    // TODO: fill this function
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
        if(node->name == "nn.batch_norm2d") {
            found = 1;
            break;
        }
    }
    if(found == 0) {
        return;
    }
    // 1. 创建新计算图，使其batch_size为输入数据集的batch_size
    std::string new_graph_content;
    std::vector<std::string> graph_lines = split(graph_content, "\n");
    for(std::string graph_line: graph_lines) {
        // 遍历旧计算图的每一行，如果是Input算子，则修改其batch_size
        std::string new_graph_line;         // 要存入新计算图的算子行
        graph_line = replace(graph_line, " ", "");
        // 提取算子名称
        int index = 0;
        while(graph_line[index] != '=') {
            index++;
        }
        index++;
        std::string op_name;
        while(graph_line[index] != '(') {
            op_name.push_back(graph_line[index]);
            index++;
        }
        if(op_name == "input") {
            // 如果是input算子，首先找到shape，之后修改batch_size
            index = (int)graph_line.find("shape");          // 找到shape
            index += 7;                                     // 越过shape=(，指向batch_size的位置
            for(int i = 0; i<index; i++) {
                new_graph_line.push_back(graph_line[i]);    // 将shape的batch_size之前的内容存入new_graph_line
            }
            // 向new_graph_line写入合适的batch_size
            new_graph_line += std::to_string(calc_running_img->size[0]);
            while(graph_line[index] != ',') {               // 找到batch_size之后的逗号
                index++;
            }
            for(; index < (int)graph_line.size(); index++) {
                new_graph_line.push_back(graph_line[index]);
            }
        }
        else {
            new_graph_line = graph_line;
        }
        // 将修改后的行加入new_graph_content
        new_graph_content += new_graph_line;
        new_graph_content.push_back('\n');
    }
    Graph new_graph{new_graph_content};
    // 2. 使用新计算图进行一次前向传播计算
    new_graph.alloc_intermediate_results();
    new_graph.forward(calc_running_img);
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
        if(this->node_list[layer]->name != "nn.batch_norm2d") {     // 跳过不是bn的层
            continue;
        }
        int bn_input_node = ((Batch_Norm2d*)this->node_list[layer]->op)->input_node; // bn层输入节点编号
        if(this->node_list[bn_input_node]->name != "nn.conv2d") {
            // 检查bn是否在conv2d之后
            fprintf(stderr, "File graph.cpp, line %d. Only bn after conv2d allowed\n", __LINE__);
            exit(-1);
        }
        // 此时layer指向bn层，bn_input_node指向conv2d层
        // 使用new_graph计算running_mean running_var
        Tensor<float32> running_mean = ((Tensor<float32>*)new_graph.intermediate_results[bn_input_node])->
                mean(std::vector<int>{0,2,3});
        Tensor<float32> running_var = ((Tensor<float32>*)new_graph.intermediate_results[bn_input_node])->
                var(std::vector<int>{0,2,3});
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
        if(i->name == "nn.batch_norm2d") {
            continue;
        }
        // 否则修改其number为上一个算子的number+1
        i->number = (int)node_list.size();
        // 向上查找当前节点的输入节点
        Node* input_node1 = nullptr;
        Node* input_node2 = nullptr;
        if(i->name == "nn.conv2d") {
            input_node1 = new_graph.node_list[((Conv2d*)i->op)->input_node];
        }
        else if(i->name == "input") {
            // do nothing
        }
        else if(i->name == "nn.relu") {
            input_node1 = new_graph.node_list[((Relu*)i->op)->input_node];
        }
        else if(i->name == "nn.maxpool2d") {
            input_node1 = new_graph.node_list[((Maxpool2d*)i->op)->input_node];
        }
        else if(i->name == "nn.flatten") {
            input_node1 = new_graph.node_list[((Flatten*)i->op)->input_node];
        }
        else if(i->name == "nn.dense") {
            input_node1 = new_graph.node_list[((Dense*)i->op)->input_node];
        }
        else if(i->name == "add") {
            input_node1 = new_graph.node_list[((Add*)i->op)->input_node1];
            input_node2 = new_graph.node_list[((Add*)i->op)->input_node2];
        }
        else if(i->name == "concat") {
            input_node1 = new_graph.node_list[((Concat*)i->op)->input_node1];
            input_node2 = new_graph.node_list[((Concat*)i->op)->input_node2];
        }
        else if(i->name == "output") {
            input_node1 = new_graph.node_list[((Output*)i->op)->input_node];
        }
        // 如果是bn，则再向上查找一个输入节点
        if(input_node1 != nullptr && input_node1->name == "nn.batch_norm2d") {
            input_node1 = new_graph.node_list[((Batch_Norm2d*)input_node1->op)->input_node];
        }
        if(input_node2 != nullptr && input_node2->name == "nn.batch_norm2d") {
            input_node2 = new_graph.node_list[((Batch_Norm2d*)input_node2->op)->input_node];
        }
        // 计数input_node1和input_node2之前删除的节点数量(即input_node之前的bn数量)
        int count1 = 0;
        if(input_node1 != nullptr) {
            for (int idx = 0; idx < input_node1->number; idx++) {
                if (new_graph.node_list[idx]->name == "nn.batch_norm2d") {
                    count1++;
                }
            }
        }
        int count2 = 0;
        if(input_node2 != nullptr) {
            for(int idx = 0; idx < input_node2->number; idx++) {
                if(new_graph.node_list[idx]->name == "nn.batch_norm2d") {
                    count2++;
                }
            }
        }
        // 用输入节点编号减去删除节点数量
        if(i->name == "nn.conv2d") {
            ((Conv2d*)i->op)->input_node = input_node1->number - count1;
        }
        else if(i->name == "input") {
            // do nothing
        }
        else if(i->name == "nn.relu") {
            ((Relu*)i->op)->input_node = input_node1->number - count1;
        }
        else if(i->name == "nn.maxpool2d") {
            ((Maxpool2d*)i->op)->input_node = input_node1->number - count1;
        }
        else if(i->name == "nn.flatten") {
            ((Flatten*)i->op)->input_node = input_node1->number - count1;
        }
        else if(i->name == "nn.dense") {
            ((Dense*)i->op)->input_node = input_node1->number - count1;
        }
        else if(i->name == "add") {
            ((Add*)i->op)->input_node1 = input_node1->number - count1;
            ((Add*)i->op)->input_node2 = input_node2->number - count2;
        }
        else if(i->name == "concat") {
            ((Concat*)i->op)->input_node1 = input_node1->number - count1;
            ((Concat*)i->op)->input_node2 = input_node2->number - count2;
        }
        else if(i->name == "output") {
            ((Output*)i->op)->input_node = input_node1->number - count1;
        }
        // 将修改后的节点加入节点列表
        node_list.push_back(i);
    }
    // 释放new_graph的中间结果
    new_graph.free_intermediate_results();
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
        if(node->name == "output") {
            ret.push_back(intermediate_results[node->number]);
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



