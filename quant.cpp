

#include <cstdio>
#include <iostream>
#include <string>

#include "graph.h"
#include "arg_handle.h"
#include "tensor.h"
#include "preprocess.h"

System_info * sys_info;

int main(int argc, char *argv[]) {
    /////////////// test
//    Tensor<int32> A{std::vector<int>{4}};
//    A.set_rand();
//    A.print();
//    Tensor<float32> B = A.astype_float32();
//    B.print();
//    return 0;
    ///////////////////////////

    sys_info = new System_info();       // 读取一些系统信息

    Graph * graph = nullptr;                                    // 计算图
    std::string calib_set_path;                                 // calibration set 路径
    int calib_size[4];                                          // calibration尺寸
    Tensor<uint8>* calib_set = nullptr;                        // calibration set
//    bool calc_running = false;                                  // 是否现场计算running
//    int running_size;                                           // running尺寸
    Tensor<uint8>* calc_running_img = nullptr;                 // 计算running数据集
//    Tensor<float32>* running_mean = nullptr;                   // running mean
//    Tensor<float32>* running_var = nullptr;                    // running var

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
        else if(option == "--calib_set") {  // 读取包含calib set路径的txt文件路径
            calib_set_path = value;
        }
        else if(option == "--calib_size") { // 读取calib尺寸
            get_calib_size(calib_size, value);
        }
//        else if(option == "--calc_running") {   // 读取是否现场计算running
//            calc_running = string_to_bool(value);
//        }
//        else if(option == "--running_size") {   // 读取running大小
//            running_size = get_running_size(value);
//        }
        else if(option == "--calc_running_img_list") {  // 读取计算running数据集
            calc_running_img = get_calc_running_img(value, calib_size);
        }
//        else if(option == "--running_mean_var_binary") {    // 读取二进制running数据
//            get_running_mean_var_binary(running_mean, running_var, value, running_size);
//        }
//        else if(option == "--running_mean_var_txt") {       // 读取txt running数据
//            get_running_mean_var_txt(running_mean, running_var, value, running_size);
//        }
        else {
            std::cerr << "option " << option << " not allowed\n";
        }
    }

    calib_set = get_calib_set(calib_set_path, calib_size);  // 读取 calibration set
    // 目前，计算图已经生成
    // calibration set，calibration尺寸 ，bn数据集(或bn数据)已经获得

    // TODO: 数据预处理
    Tensor<float32>* processed_calib_set = preprocess(calib_set);
    Tensor<float32>* processed_bn_set = preprocess(calc_running_img);

    // TODO: fuse operator
    graph->fuse_op(calc_running_img);

    // TODO: quantization

    // TODO: save quantized model



    delete(graph);
    delete(calib_set);
    delete(calc_running_img);
//    delete(running_mean);
//    delete(running_var);
    delete(processed_calib_set);
    delete(processed_bn_set);

    return 0;
}
