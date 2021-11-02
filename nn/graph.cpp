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
    Vdarray<float32> * temp = new Vdarray<float32>{std::vector<int>{1, 10}};
    temp->set_rand();
    std::vector<void*> ret;
    ret.push_back(temp);
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

void test_accuracy(const std::string &val_set_path, Graph *graph, int *infer_shape) {
    /*
     * 测试计算图准确率
     * 只用于分类任务
     */
    printf("Test accuracy:\n");
    // 打开val set数据集文件
    std::ifstream file;
    file.open(val_set_path, std::ios::in);
    if(!file.is_open()) {
        std::cerr << "calib set txt file not found\n";
        exit(-1);
    }
    // 遍历文件里的每一行
    int correct = 0;
    int total = 0;
    std::string line;
    graph->alloc_intermediate_results();    // 为中间结果分配内存
    while(std::getline(file, line)) {
        // 将行分为图片路径和分类标签
        std::vector<std::string> line_split = split(line, " ");
        std::string img_path = line_split[0];
        int answer = (int)strtol(line_split[1].c_str(), nullptr, 10);
        // 将img读入Vdarray
        cv::Mat img;
        cv::Mat dst;
        if(infer_shape[1] == 1) {
            img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);;
        }
        else if(infer_shape[1] == 3) {
            img = cv::imread(img_path, cv::IMREAD_COLOR);;
        }
        // Resize
        cv::resize(img, dst, cv::Size(infer_shape[3], infer_shape[2]), 0, 0, cv::INTER_LINEAR);
        // 存储resized图片到Vdarray
        Vdarray<uint8> bgr_hwc_img(std::vector<int>{infer_shape[2], infer_shape[3], infer_shape[1]});
        memcpy(bgr_hwc_img.data, dst.data, sizeof(unsigned char)*infer_shape[2]*infer_shape[3]*infer_shape[1]);
        // hwc to chw
        Vdarray<uint8> bgr_chw_img = bgr_hwc_img.transpose(std::vector<int>{2, 0, 1});
        // bgr to rgb
        Vdarray<uint8> rgb_chw_img(bgr_chw_img.shape());
        if(infer_shape[1] == 3) {
            rgb_chw_img[0] = bgr_chw_img[2];
            rgb_chw_img[1] = bgr_chw_img[1];
            rgb_chw_img[2] = bgr_chw_img[0];
        }
        else {
            rgb_chw_img[0] = bgr_chw_img[0];
        }
        // 调用graph->forward
        // 不需要释放result_vector中的结果, 应为它们是指向graph中intermediate_results里空间的指针，在forward返回时不会分配新空间
        std::vector<void*> result_vector = graph->forward(&rgb_chw_img);
        int result = ((Vdarray<float32>*)(result_vector[0]))->argmax();
        if(result == answer) {
            correct ++;
        }
        total ++;
        printf("\rProcessing: %d", total);
        fflush(stdout);
    }
    printf("Correct: %d, Total: %d, accuracy: %f\n", correct, total, (float)correct/(float)total);
    graph->free_intermediate_results();
}
