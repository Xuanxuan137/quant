

#include <cstdio>
#include <iostream>
#include <string>
#include <quant_tools.h>


#include "graph.h"
#include "arg_handle.h"
#include "tensor.h"
#include "preprocess.h"

void test_accuracy(const std::string &val_set_path, Graph *graph, int *infer_shape);
void test_quant_accuracy(const std::string &val_set_path, Graph *graph, int *infer_shape);

System_info * sys_info;

int main(int argc, char *argv[]) {
    /////////////// test

    ///////////////////////////

    sys_info = new System_info();       // 读取一些系统信息

    Graph * graph = nullptr;                                    // 计算图
    std::string calib_set_path;                                 // calibration set 路径
    int calib_size[4];                                          // calibration尺寸
    Tensor<uint8>* calib_set = nullptr;                        // calibration set
    Tensor<uint8>* calc_running_img = nullptr;                 // 计算running数据集
    std::string output_dir;                                     // 输出路径

    for(int i = 1; i<argc; i++) {
        std::string option(argv[i]);    // 从argv读取选项
        i++;
        if(i >= argc) {
            std::cerr << "Got none value after option " << option << std::endl;
            return -1;
        }
        std::string value(argv[i]);     // 选项值

        // 处理选项
        if(option == "--graph") {           // 创建计算图
            printf("Reading calculation graph...\n");
            // 将计算图文件里的内容读取到字符串里
            // 打开计算图文件
            std::ifstream graph_file;
            graph_file.open(value, std::ios::in);
            if(!graph_file.is_open()) {
                std::cerr << "calib set txt file not found\n";
                exit(-1);
            }
            // 将有信息的行加入graph_content
            std::string graph_content;
            std::string graph_line;
            while(std::getline(graph_file, graph_line)) {
                graph_line = delete_annotation(graph_line, "#");
                graph_line = replace(graph_line, " ", "");
                if(graph_line.empty()) {
                    continue;
                }
                graph_content += graph_line;
                graph_content.push_back('\n');
            }
            graph = new Graph(graph_content);
            printf("Read calculation graph finished\n");
        }
        else if(option == "--calib_set") {  // 读取包含calib set路径的txt文件路径
            calib_set_path = value;
        }
        else if(option == "--calib_size") { // 读取calib尺寸
            get_calib_size(calib_size, value);
        }
        else if(option == "--calc_running_img_list") {  // 读取计算running数据集
            calc_running_img = get_calc_running_img(value, calib_size);
        }
        else if(option == "--output_dir") {     // 读取输出路径
            output_dir = value;
        }
        else {
            std::cerr << "option " << option << " not allowed\n";
        }
    }
    if(graph == nullptr) {
        fprintf(stderr, "--graph is required\n");
        exit(-1);
    }

    calib_set = get_calib_set(calib_set_path, calib_size);  // 读取 calibration set
    // 目前，计算图已经生成
    // calibration set，calibration尺寸 ，bn数据集(或bn数据)已经获得

    // 数据预处理
    Tensor<float32>* processed_calib_set = preprocess(calib_set);
    Tensor<float32>* processed_bn_set = preprocess(calc_running_img);

    // fuse operator
    graph->fuse_op(processed_bn_set);   // 如果计算图中不包含bn，会自动跳过此步骤
    int infer_shape[4] = {1,1,28,28};
    test_accuracy("../val_set.txt", graph, infer_shape);

    // quantization
    Graph * q_graph = graph->quantization(calib_set, processed_calib_set);
    test_quant_accuracy("../val_set.txt", q_graph, infer_shape);

    // save quantized model
    q_graph->save(output_dir);


    delete(graph);
    delete(calib_set);
    delete(calc_running_img);
    delete(processed_calib_set);
    delete(processed_bn_set);
    delete(q_graph);

    return 0;
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
        // 将img读入Tensor
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
        // 存储resized图片到Tensor
        Tensor<uint8> bgr_hwc_img(std::vector<int>{infer_shape[2], infer_shape[3], infer_shape[1]});
        memcpy(bgr_hwc_img.data, dst.data, sizeof(unsigned char)*infer_shape[2]*infer_shape[3]*infer_shape[1]);
        // hwc to chw
        Tensor<uint8> bgr_chw_img = bgr_hwc_img.transpose(std::vector<int>{2, 0, 1});
        // bgr to rgb
        Tensor<uint8> rgb_chw_img(bgr_chw_img.shape());
        if(infer_shape[1] == 3) {
            rgb_chw_img[0] = bgr_chw_img[2];
            rgb_chw_img[1] = bgr_chw_img[1];
            rgb_chw_img[2] = bgr_chw_img[0];
        }
        else {
            rgb_chw_img[0] = bgr_chw_img[0];
        }
        rgb_chw_img = rgb_chw_img.reshape(std::vector<int>{1, rgb_chw_img.size[0], rgb_chw_img.size[1], rgb_chw_img.size[2]});
        // 根据graph类型进行数据预处理
        void * processed_input = nullptr;
        if(graph->node_list[0]->dtype == "uint8") {
            processed_input = &rgb_chw_img;
        }
        else {
            processed_input = preprocess(&rgb_chw_img);
        }
        // 调用graph->forward
        // 不需要释放result_vector中的结果, 应为它们是指向graph中intermediate_results里空间的指针，在forward返回时不会分配新空间
        std::vector<void*> result_vector = graph->forward(processed_input);
        // 释放processed_input
        if(graph->node_list[0]->dtype == "uint8") {
            delete((Tensor<uint8>*)processed_input);
        }
        else {
            delete((Tensor<float32>*)processed_input);
        }
        int result = ((Tensor<float32>*)(result_vector[0]))->argmax();
        if(result == answer) {
            correct ++;
        }
        total ++;
        printf("\rProcessing: %d. Correct: %d, accuracy: %f", total, correct, (float)correct/(float)total);
        fflush(stdout);
    }
    printf("\n");
    printf("Correct: %d, Total: %d, accuracy: %f\n", correct, total, (float)correct/(float)total);
    graph->free_intermediate_results();
}


uint8 image[784] = { 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,  84, 185, 159, 151,  60,  36,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0, 222, 254, 254, 254, 254, 241, 198, 198, 198, 198, 198, 198, 198, 198, 170,  52,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,  67, 114,  72, 114, 163, 227, 254, 225, 254, 254, 254, 250, 229, 254, 254, 140,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  17,  66,  14,  67,  67,  67,  59,  21, 236, 254, 106,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  83, 253, 209,  18,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  22, 233, 255,  83,   0,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 129, 254, 238,  44,   0,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  59, 249, 254,  62,   0,   0,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 133, 254, 187,   5,   0,   0,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   9, 205, 248,  58,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 126, 254, 182,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  75, 251, 240,  57,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  19, 221, 254, 166,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3, 203, 254, 219,  35,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  38, 254, 254,  77,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  31, 224, 254, 115,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 133, 254, 254,  52,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  61, 242, 254, 254,  52,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 121, 254, 254, 219,  40,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 121, 254, 207,  18,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0};

void test_quant_accuracy(const std::string &val_set_path, Graph *graph, int *infer_shape) {
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
        // 将img读入Tensor
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
        // 存储resized图片到Tensor
        Tensor<uint8> bgr_hwc_img(std::vector<int>{infer_shape[2], infer_shape[3], infer_shape[1]});
        memcpy(bgr_hwc_img.data, dst.data, sizeof(unsigned char)*infer_shape[2]*infer_shape[3]*infer_shape[1]);
        // hwc to chw
        Tensor<uint8> bgr_chw_img = bgr_hwc_img.transpose(std::vector<int>{2, 0, 1});
        // bgr to rgb
        Tensor<uint8> rgb_chw_img(bgr_chw_img.shape());
        if(infer_shape[1] == 3) {
            rgb_chw_img[0] = bgr_chw_img[2];
            rgb_chw_img[1] = bgr_chw_img[1];
            rgb_chw_img[2] = bgr_chw_img[0];
        }
        else {
            rgb_chw_img[0] = bgr_chw_img[0];
        }
        rgb_chw_img = rgb_chw_img.reshape(std::vector<int>{1, rgb_chw_img.size[0], rgb_chw_img.size[1], rgb_chw_img.size[2]});
        // 根据graph类型进行数据预处理
        void * processed_input = nullptr;
        if(graph->node_list[0]->dtype == "uint8") {
            processed_input = &rgb_chw_img;
        }
        else {
            processed_input = preprocess(&rgb_chw_img);
        }
        // 调用graph->forward
        // 不需要释放result_vector中的结果, 应为它们是指向graph中intermediate_results里空间的指针，在forward返回时不会分配新空间
//        memcpy(((Tensor<uint8>*)processed_input)->data, image, sizeof(uint8)*784);
        std::vector<void*> result_vector = graph->forward(processed_input);
        int result = ((Tensor<uint8>*)(result_vector[0]))->argmax();
        if(result == answer) {
            correct ++;
        }
        total ++;
        printf("\rProcessing: %d. Correct: %d, accuracy: %f", total, correct, (float)correct/(float)total);
        fflush(stdout);
    }
    printf("\n");
    printf("Correct: %d, Total: %d, accuracy: %f\n", correct, total, (float)correct/(float)total);
    graph->free_intermediate_results();
}

