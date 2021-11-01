//
// Created by noname on 2021/10/24.
//

#ifndef QUANT_ARG_HANDLE_H
#define QUANT_ARG_HANDLE_H

#include <iostream>
#include <string>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/hal/interface.h>

#include "vdarray.h"
#include "util.h"

void get_calib_size(int * calib_size, std::string value);
int get_running_size(const std::string &value);
Vdarray<unsigned char>* get_calib_set(const std::string& calib_set_path, int calib_size[]);
Vdarray<unsigned char>* get_calc_running_img(const std::string& running_set_path, int calib_size[]);
void get_running_mean_var_binary(Vdarray<float>* &running_mean, Vdarray<float>* &running_var,
                                 const std::string& path, int size);
void get_running_mean_var_txt(Vdarray<float>* &running_mean, Vdarray<float>* &running_var,
                              const std::string& path, int size);


#endif //QUANT_ARG_HANDLE_H
