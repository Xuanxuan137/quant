//
// Created by noname on 2021/10/23.
//

#include "graph.h"


Graph::Graph(const std::string& path)
{
    /*
     * 读取path的文件，根据它创建计算图
     * 每读取一行，创建一个节点
     */
    printf("Reading calculation graph...\n");

    // 打开计算图文件
    std::ifstream graph_file;
    graph_file.open(path, std::ios::in);
    if(!graph_file.is_open()) {
        std::cerr << "calib set txt file not found\n";
        exit(-1);
    }
    // 根据每行创建一个节点
    std::string graph_line;
    while(std::getline(graph_file, graph_line)) {
        if(replace(graph_line, " ", "") == "\n") {
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

void Graph::fuse_op(bool calc_running, int running_size, Vdarray<uint8> *calc_running_img,
               Vdarray<float32> *running_mean, Vdarray<float32> *running_var)
{
    // TODO: fill this function
}

std::vector<void*> Graph::forward(void *input) {
    /*
     * 前向传播函数：
     * 输入和返回类型实际均为Vdarray<>*
     * 该函数应使用intermediate_results存储中间结果，且intermediate_results的实际类型为std::vector<Vdarray<>*>
     * 由于Graph不管理类型，所有Vdarray<>*均设为void*。调用者必须负责管理类型
     *
     * 计算方式:
     * 1. 不需要初始化intermediate_results。intermediate_results应由调用者管理
     * 2. 遍历graph中所有节点，传入input和output指针
     * 3. 如果某个节点是output节点，那么将它对应的中间结果Vdarray数组指针加入一个vector，并最终返回这个vector
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
            Vdarray<float32> *inter_res = new Vdarray<float32>{node->output_shape};
            intermediate_results.push_back(inter_res);
        }
        else if(node->dtype == "uint8") {
            Vdarray<uint8> *inter_res = new Vdarray<uint8>{node->output_shape};
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
            delete((Vdarray<float32>*)intermediate_results[i]);
        }
        if(node_list[i]->dtype == "uint8") {
            delete((Vdarray<uint8>*)intermediate_results[i]);
        }
    }
    while(!intermediate_results.empty()) {
        intermediate_results.pop_back();
    }
}


