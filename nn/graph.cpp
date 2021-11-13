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
    printf("Reading calculation graph...\n");
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
    // 0. 检查是否包含bn
    int found = 0;
    for(Node *node: node_list) {
        if(node->name == OPN_BATCH_NORM2D) {
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
    for(int layer = 0; layer < (int)this->node_list.size(); layer++) {
        if(this->node_list[layer]->name != OPN_BATCH_NORM2D) {
            
        }
    }



    new_graph.free_intermediate_results();
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



