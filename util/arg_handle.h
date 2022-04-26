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

#include "tensor.h"
#include "util.h"


Tensor<unsigned char>* get_calib_set(const std::string& calib_set_path, 
                                     const std::vector<int>& calib_size);


#endif //QUANT_ARG_HANDLE_H
