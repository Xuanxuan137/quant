//
// Created by noname on 2021/11/3.
//


/*
 * 测试模型准确率(未量化)
 * 需要：
 * --graph
 * --val_set
 * --infer_shape
 */


#include <cstdio>
#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/hal/interface.h>

#include "tensor.h"
#include "util.h"
#include "preprocess.h"
#include "graph.h"
#include "arg_handle.h"


void test_accuracy(const std::string &val_set_path, Graph * graph, int *infer_shape);   // 根据val set路径测试准确率

int main(int argc, char *argv[]) {

    Graph * graph = nullptr;                                    // 计算图
    std::string val_set_path;                                   // calibration set 路径
    int infer_shape[4];                                         // calibration尺寸

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
            graph = new Graph(value);
        }
        else if(option == "--val_set") {  // 读取包含calib set路径的txt文件路径
            val_set_path = value;
        }
        else if(option == "--infer_shape") { // 读取calib尺寸
            get_calib_size(infer_shape, value);
        }
        else {
            std::cerr << "option " << option << " not allowed\n";
        }
    }

    // 目前，计算图已经生成
    unsigned long long start_time = get_micro_sec_time();
    test_accuracy(val_set_path, graph, infer_shape);
    unsigned long long end_time = get_micro_sec_time();
    printf("time: %llu\n", end_time-start_time);



    delete(graph);

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