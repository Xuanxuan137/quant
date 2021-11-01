//
// Created by noname on 2021/10/24.
//

#include "arg_handle.h"


void get_calib_size(int * calib_size, std::string value)
{
    /*
     * Read calibration set size from input argument. For instance, 1,3,224,224
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
     * Read running_mean_var size. For instance, 16
     */
    return (int)strtol(value.c_str(), nullptr, 10);
}


Vdarray<unsigned char>* get_calib_set(const std::string& calib_set_path, int calib_size[])
{
    /*
     * Open txt file which record the path of calibration set, and read the path in it.
     * Create Vdarray object pointer with these pictures, and return this pointer.
     */
    std::cout << "Reading calib set...\n";

    std::ifstream file;
    file.open(calib_set_path, std::ios::in);
    if(!file.is_open()) {
        std::cerr << "calib set txt file not found\n";
        exit(-1);
    }
    int img_num = 0;        // Count the number of pictures
    std::string img_path;
    while(std::getline(file, img_path)) {
        if(replace(img_path, " ", "") == "\n") {
            continue;
        }
        img_num++;
    }
    // Reopen the file and bring the file pointer back to the beginning(it seems that 'seekg()' does not work)
    file.close();
    file.open(calib_set_path, std::ios::in);
    if(!file.is_open()) {
        std::cerr << "calib set txt file not found\n";
        exit(-1);
    }

    // Read pictures and save into calib_set
    Vdarray<unsigned char> *calib_set = new Vdarray<unsigned char>(std::vector<int>{img_num, calib_size[1], calib_size[2], calib_size[3]});   // create calib_set space
    int count = 0;
    while(std::getline(file, img_path)) {   // read path of a picture
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
        // resize original picture to input shape
        cv::resize(img, dst, cv::Size(calib_size[3], calib_size[2]), 0, 0, cv::INTER_LINEAR);
        // save resized picture into Vdarray
        Vdarray<unsigned char> bgr_hwc_img(std::vector<int>{calib_size[2], calib_size[3], calib_size[1]});
        memcpy(bgr_hwc_img.data, dst.data, sizeof(unsigned char)*calib_size[2]*calib_size[3]*calib_size[1]);
        // hwc to chw
        Vdarray<unsigned char> bgr_chw_img = bgr_hwc_img.transpose(std::vector<int>{2, 0, 1});
        // bgr to rgb
        Vdarray<unsigned char> rgb_chw_img(bgr_chw_img.shape());
        if(calib_size[1] == 3) {
            rgb_chw_img[0] = bgr_chw_img[2];
            rgb_chw_img[1] = bgr_chw_img[1];
            rgb_chw_img[2] = bgr_chw_img[0];
        }
        else {
            rgb_chw_img[0] = bgr_chw_img[0];
        }
        // save into calib_set
        (*calib_set)[count] = rgb_chw_img;
        count++;
        printf("\r%d/%d", count, img_num);
        fflush(stdout);
    }
    printf("\rRead calib set finished\n");
    return calib_set;
}


Vdarray<unsigned char>* get_calc_running_img(const std::string& running_set_path, int calib_size[])
{
    /*
     * Open txt file which records the path of the dataset used to calculate running_mean_var, and
     * read the path in it.
     * Create Vdarray object pointer with these pictures, and return this pointer.
     */
    std::cout << "Reading running set...\n";

    std::ifstream file;
    file.open(running_set_path, std::ios::in);
    if(!file.is_open()) {
        std::cerr << "calib set txt file not found\n";
        exit(-1);
    }
    int img_num = 0;        // Count number of pictures
    std::string img_path;
    while(std::getline(file, img_path)) {
        if(replace(img_path, " ", "") == "\n") {
            continue;
        }
        img_num++;
    }
    // Reopen the file and bring the file pointer back to the beginning(it seems that 'seekg()' does not work)
    file.close();
    file.open(running_set_path, std::ios::in);
    if(!file.is_open()) {
        std::cerr << "calib set txt file not found\n";
        exit(-1);
    }

    // Read pictures and save into running_set
    Vdarray<unsigned char> *running_set = new Vdarray<unsigned char>(std::vector<int>{img_num, calib_size[1], calib_size[2], calib_size[3]});   // create dataset space
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
        // Resize original picture to input shape
        cv::resize(img, dst, cv::Size(calib_size[3], calib_size[2]), 0, 0, cv::INTER_LINEAR);
        // Save resized picture into Vdarray
        Vdarray<unsigned char> bgr_hwc_img(std::vector<int>{calib_size[2], calib_size[3], calib_size[1]});
        memcpy(bgr_hwc_img.data, dst.data, sizeof(unsigned char)*calib_size[2]*calib_size[3]*calib_size[1]);
        // hwc to chw
        Vdarray<unsigned char> bgr_chw_img = bgr_hwc_img.transpose(std::vector<int>{2, 0, 1});
        // bgr to rgb
        Vdarray<unsigned char> rgb_chw_img(bgr_chw_img.shape());
        if(calib_size[1] == 3) {
            rgb_chw_img[0] = bgr_chw_img[2];
            rgb_chw_img[1] = bgr_chw_img[1];
            rgb_chw_img[2] = bgr_chw_img[0];
        }
        else {
            rgb_chw_img[0] = bgr_chw_img[0];
        }
        // save into calib_set
        (*running_set)[count] = rgb_chw_img;
        count++;
        printf("\r%d/%d", count, img_num);
        fflush(stdout);
    }
    printf("\rRead calib set finished\n");
    return running_set;
}

void get_running_mean_var_binary(Vdarray<float>* &running_mean, Vdarray<float>* &running_var,
                                 const std::string& path, int size)
{
    /*
     * Allocate space for running_mean_var according to size
     * Read binary float value from file in path, and save into running_mean and running_var in turn
     */
    running_mean = new Vdarray<float>(std::vector<int>{size});
    running_var = new Vdarray<float>(std::vector<int>{size});
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

void get_running_mean_var_txt(Vdarray<float>* &running_mean, Vdarray<float>* &running_var,
                              const std::string& path, int size)
{
    /*
     * Allocate space for running_mean_var according to size
     * Read txt float value from file in path, and save into running_mean and running_var in turn
     */
    std::ifstream file;
    file.open(path, std::ios::in);
    if (!file.is_open()) {
        std::cerr << "calib set txt file not found\n";
        exit(-1);
    }

    running_mean = new Vdarray<float>(std::vector<int>{size});
    running_var = new Vdarray<float>(std::vector<int>{size});
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

