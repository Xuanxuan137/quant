//
// Created by noname on 2021/10/24.
//

#include "arg_handle.h"


void get_calib_size(int * calib_size, std::string value)
{
    /*
     * 读取calib尺寸，如1,3,224,224
     */
    value = replace(value, " ", "");
    std::vector<std::string> sizes = split(value, ",");
    assert(sizes.size() == 4);

    for(int i = 0; i<4; i++) {
        calib_size[i] = (int)strtol(sizes[i].c_str(), nullptr, 10);
    }
}


int get_running_size(const std::string &value) {
    /*
     * 读取running mean var大小，如16
     */
    return (int)strtol(value.c_str(), nullptr, 10);
}


Tensor<unsigned char>* get_calib_set(const std::string& calib_set_path, int calib_size[])
{
    /*
     * 打开包含calib set数据集路径的txt文件，读取里面的路径
     * 使用这些图片创建Tensor对象，并返回指针
     */
    std::cout << "Reading calib set...\n";

    std::ifstream file;
    file.open(calib_set_path, std::ios::in);
    if(!file.is_open()) {
        std::cerr << "calib set txt file not found\n";
        exit(-1);
    }
    int img_num = 0;        // 计算图片数量
    std::string img_path;
    while(std::getline(file, img_path)) {
        if(replace(img_path, " ", "").empty()) {
            continue;
        }
        img_num++;
    }
    // 重新打开file，使文件指针回到开头(seekg()似乎无效)
    file.close();
    file.open(calib_set_path, std::ios::in);
    if(!file.is_open()) {
        std::cerr << "calib set txt file not found\n";
        exit(-1);
    }

    // 读取图片，存入calib_set
    Tensor<unsigned char> *calib_set = new Tensor<unsigned char>(std::vector<int>{img_num, calib_size[1], calib_size[2], calib_size[3]});   // create calib_set space
    int count = 0;
    while(std::getline(file, img_path)) {   // 读取一个图片路径
        cv::Mat img;
        if(calib_size[1] == 1) {
            img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);;
        }
        else if(calib_size[1] == 3) {
            img = cv::imread(img_path, cv::IMREAD_COLOR);;
        }
        else {
            std::cerr << "Channel of calib_size can only be 1 or 3\n";
            exit(-1);
        }
        cv::Mat dst;
        // resize为输入的尺寸
        cv::resize(img, dst, cv::Size(calib_size[3], calib_size[2]), 0, 0, cv::INTER_LINEAR);
        // 将resized图片存入Tensor
        Tensor<unsigned char> bgr_hwc_img(std::vector<int>{calib_size[2], calib_size[3], calib_size[1]});
        memcpy(bgr_hwc_img.data, dst.data, sizeof(unsigned char)*calib_size[2]*calib_size[3]*calib_size[1]);
        // hwc to chw
        Tensor<unsigned char> bgr_chw_img = bgr_hwc_img.transpose(std::vector<int>{2, 0, 1});
        // bgr to rgb
        Tensor<unsigned char> rgb_chw_img(bgr_chw_img.shape());
        if(calib_size[1] == 3) {
            rgb_chw_img[0] = bgr_chw_img[2];
            rgb_chw_img[1] = bgr_chw_img[1];
            rgb_chw_img[2] = bgr_chw_img[0];
        }
        else {
            rgb_chw_img[0] = bgr_chw_img[0];
        }
        // 存入calib_set
        (*calib_set)[count] = rgb_chw_img;
        count++;
        printf("\r%d/%d", count, img_num);
        fflush(stdout);
    }
    printf("\rRead calib set finished\n");
    return calib_set;
}


Tensor<unsigned char>* get_calc_running_img(const std::string& running_set_path, int calib_size[])
{
    /*
     * 打开包含bn数据集路径的txt文件，并读取里面的路径
     * 根据这些图片创建Tensor，并返回指针
     */
    std::cout << "Reading running set...\n";

    std::ifstream file;
    file.open(running_set_path, std::ios::in);
    if(!file.is_open()) {
        std::cerr << "calib set txt file not found\n";
        exit(-1);
    }
    int img_num = 0;        // 计算图片数量
    std::string img_path;
    while(std::getline(file, img_path)) {
        if(replace(img_path, " ", "").empty()) {
            continue;
        }
        img_num++;
    }
    // 重新打开file，使文件指针回到开头(seekg()似乎无效)
    file.close();
    file.open(running_set_path, std::ios::in);
    if(!file.is_open()) {
        std::cerr << "calib set txt file not found\n";
        exit(-1);
    }

    // 读取图片，存入running_set
    Tensor<uint8> *running_set = new Tensor<unsigned char>(std::vector<int>{img_num, calib_size[1], calib_size[2], calib_size[3]});   // create dataset space
    int count = 0;
    while(std::getline(file, img_path)) {   // read a picture path
        cv::Mat img;
        if(calib_size[1] == 1) {
            img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);;
        }
        else if(calib_size[1] == 3) {
            img = cv::imread(img_path, cv::IMREAD_COLOR);;
        }
        else {
            std::cerr << "Channel of calib_size can only be 1 or 3\n";
            exit(-1);
        }
        cv::Mat dst;
        // Resize图片为输入尺寸
        cv::resize(img, dst, cv::Size(calib_size[3], calib_size[2]), 0, 0, cv::INTER_LINEAR);
        // 存储resized图片到Tensor
        Tensor<uint8> bgr_hwc_img(std::vector<int>{calib_size[2], calib_size[3], calib_size[1]});
        memcpy(bgr_hwc_img.data, dst.data, sizeof(unsigned char)*calib_size[2]*calib_size[3]*calib_size[1]);
        // hwc to chw
        Tensor<uint8> bgr_chw_img = bgr_hwc_img.transpose(std::vector<int>{2, 0, 1});
        // bgr to rgb
        Tensor<uint8> rgb_chw_img(bgr_chw_img.shape());
        if(calib_size[1] == 3) {
            rgb_chw_img[0] = bgr_chw_img[2];
            rgb_chw_img[1] = bgr_chw_img[1];
            rgb_chw_img[2] = bgr_chw_img[0];
        }
        else {
            rgb_chw_img[0] = bgr_chw_img[0];
        }
        // 存入running_set
        (*running_set)[count] = rgb_chw_img;
        count++;
        printf("\r%d/%d", count, img_num);
        fflush(stdout);
    }
    printf("\rRead calib set finished\n");
    return running_set;
}

void get_running_mean_var_binary(Tensor<float>* &running_mean, Tensor<float>* &running_var,
                                 const std::string& path, int size)
{
    /*
     * 根据size为running_mean_var分配空间
     * 读取二进制浮点文件，并存入Tensor
     */
    running_mean = new Tensor<float>(std::vector<int>{size});
    running_var = new Tensor<float>(std::vector<int>{size});
    FILE * file = fopen(path.c_str(), "r");
    if((int)fread(running_mean->data, sizeof(float), size, file) != size) {
        std::cerr << "Read size error when reading running mean\n";
        exit(-1);
    }
    if((int)fread(running_var->data, sizeof(float), size, file) != size) {
        std::cerr << "Read size error when reading running var\n";
        exit(-1);
    }
}

void get_running_mean_var_txt(Tensor<float>* &running_mean, Tensor<float>* &running_var,
                              const std::string& path, int size)
{
    /*
     * 根据size为running_mean_var分配空间
     * 读取txt浮点文件，并存入Tensor
     */
    std::ifstream file;
    file.open(path, std::ios::in);
    if (!file.is_open()) {
        std::cerr << "calib set txt file not found\n";
        exit(-1);
    }

    running_mean = new Tensor<float>(std::vector<int>{size});
    running_var = new Tensor<float>(std::vector<int>{size});
    std::string data_str;
    for (int i = 0; i < size; i++) {
        std::getline(file, data_str);
        running_mean->data[i] = strtof(data_str.c_str(), nullptr);
    }
    for (int i = 0; i < size; i++) {
        std::getline(file, data_str);
        running_var->data[i] = strtof(data_str.c_str(), nullptr);
    }
}

