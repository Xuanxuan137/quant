//
// Created by noname on 2021/11/3.
//

#include "functional.h"
#include "cblas.h"
#include "fixed_point.h"
#include "tensor.h"
#include <thread>

extern System_info * sys_info;


Tensor<float32>
functional::conv2d(Tensor<float32> *input, Tensor<float32> *weight, Tensor<float32> *bias, const std::vector<int>& stride,
                   const std::vector<int>& padding_size, const std::vector<int>& dilation)
{
    /*
     * Conv2d
     */
    // 对输入尺寸进行校验
    if(input->size.size() != 4) {
        fprintf(stderr, "only 4 dimension input is allowed in conv2d\n");
        exit(-1);
    }
    if(weight->size.size() != 4) {
        fprintf(stderr, "only 4 dimension weight is allowed in conv2d\n");
        exit(-1);
    }
    if(input->size[1] != weight->size[1]) {
        fprintf(stderr, "channel of input should equal to input channel of weight\n");
        exit(-1);
    }
    if(bias->size.size() != 1) {
        fprintf(stderr, "only 1 dimension bias is allowed in conv2d\n");
        exit(-1);
    }
    if(bias->size[0] != weight->size[0]) {
        fprintf(stderr, "dim 1 of bias should equal to dim 1 of weight\n");
    }
    if(stride.size() != 2) {
        fprintf(stderr, "stride should be a vector of 2 int\n");
        exit(-1);
    }
    if(padding_size.size() != 2) {
        fprintf(stderr, "padding should be a vector of 2 int\n");
        exit(-1);
    }
    if(dilation.size() != 2) {
        fprintf(stderr, "dilation should be a vector of 2 int\n");
        exit(-1);
    }
    // padding
    Tensor<float32> padded = padding(input, padding_size);
    // 计算输出尺寸
    std::vector<int> kernel_size{weight->size[2], weight->size[3]};
    int batch_size = padded.size[0];
    int channel = weight->size[0];
    int height = (padded.size[2] - (dilation[0] * (kernel_size[0]-1) + 1)) / stride[0] + 1;
    int width = (padded.size[3] - (dilation[1] * (kernel_size[1]-1) + 1)) / stride[1] + 1;
    // 创建返回对象
    Tensor<float32> result{std::vector<int>{batch_size, channel, height, width}};
    // 计算conv2d
    for(int n = 0; n<batch_size; n++) {     // 每次计算1张
        // 1. 展开为矩阵
        // 1.1 weight展开为矩阵
        Tensor<float32> weight_matrix{
                std::vector<int>{weight->size[0], weight->size[1] * weight->size[2] * weight->size[3]}
        };  // OIHW:   (O) * (I*H*W)
        memcpy(weight_matrix.data, weight->data, sizeof(float32) * weight_matrix.len());  // 似乎展开之后数据顺序不变，那么直接复制过来就好啦
        // 1.2 input展开为矩阵
        Tensor<float32> temp_padded = padded[n];   // 取出input中的第n张图片
        Tensor<float32> input_matrix{
                std::vector<int>{padded.size[1] * kernel_size[0] * kernel_size[1], height * width}
        };  // NCHW:    (C*KH*KW) * (OH*OW)
        int start_h = 0;    // 从input取数时，h方向上的起点。每次增加stride_h
        for (int oh = 0; oh < height; oh++, start_h += stride[0]) {
            int start_w = 0;
            for (int ow = 0; ow < width; ow++, start_w += stride[1]) {
                for (int c = 0; c < padded.size[1]; c++) {
                    int start_kh = 0;   // 从input取数时，卷积核笼罩范围内的h方向上的起点，每次增加dilation_h
                    for (int kh = 0; kh < kernel_size[0]; kh++, start_kh += dilation[0]) {
                        int start_kw = 0;
                        for (int kw = 0; kw < kernel_size[1]; kw++, start_kw += dilation[1]) {
                            // matrix[c*kernel_size[0]*kernel_size[1] + kh*kernel_size[1] * kw][oh*width + ow] = input[c][start_h+start_kh][start_w+start_kw]
                            input_matrix.data[
                                    (c * kernel_size[0] * kernel_size[1] + kh * kernel_size[1] + kw) * height * width +
                                    (oh * width + ow)]
                                    =
                            temp_padded.data[
                                    c * padded.size[2] * padded.size[3] +
                                    (start_h + start_kh) * padded.size[3] +
                                    (start_w + start_kw)];
                        }
                    }
                }
            }
        }
        // 2. 矩阵相乘
        // Tensor<float32> result_matrix = weight_matrix.dot(input_matrix);
        // 改用openblas
        Tensor<float32> result_matrix(std::vector<int>{weight_matrix.size[0], input_matrix.size[1]});
        result_matrix.set_zero();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            weight_matrix.size[0], input_matrix.size[1], weight_matrix.size[1],
            1.0f, weight_matrix.data, weight_matrix.size[1], 
            input_matrix.data, input_matrix.size[1], 1.0f,
            result_matrix.data, result_matrix.size[1]);
        // 3. +bias
        for(int c = 0; c<channel; c++) {
            result_matrix[c] += (*bias)[c].to_num();
        }
        // 4. 变回4维结构
        memcpy(result[n].data, result_matrix.data, sizeof(float32)*result_matrix.len());
    }
    return result;
}

void mt_relu(float32 * R, float32 * I, int len)
{
    /*
     * 多线程relu的子线程
     */
    int new_len = len / 4 * 4;
    for(int i = 0; i<len; i+=4) {
        R[i] = (I[i] > 0) ? I[i] : 0;
        R[i+1] = (I[i+1] > 0) ? I[i+1] : 0;
        R[i+2] = (I[i+2] > 0) ? I[i+2] : 0;
        R[i+3] = (I[i+3] > 0) ? I[i+3] : 0;
    }
    for(int i = new_len; i<len; i++) {
        R[i] = (I[i] > 0) ? I[i] : 0;
    }
}


//Tensor<float32>
//functional::relu(Tensor<float32> *input) {
//    /*
//     * 原始Relu
//     */
//    Tensor<float32> result{input->size};
//    float32 * I = input->data;
//    float32 * R = result.data;
//
//    int len = result.len();
//    printf("len %d\n", len);
//    for(int i = 0; i<len; i++) {
//        R[i] = (I[i] > 0) ? I[i] : 0;
//    }
//    return result;
//}

Tensor<float32>
functional::relu(Tensor<float32> *input) {
    /*
     * 多线程优化Relu
     */
    Tensor<float32> result{input->size};
    float32 * I = input->data;          // 输入的数据地址
    float32 * R = result.data;          // 输出的数据地址
    int len = result.len();             // 总的要计算的元素数量

    if(len > 500000) {  // 一般大于此值，多线程才有加速效果
        int n_proc = sys_info->n_proc;      // 处理器数量
        int len_per_proc = len / n_proc;    // 每个处理器要计算的元素数量(可能由于不能整除而有剩余)
        std::thread t[n_proc];              // 创建子线程
        for (int i = 0; i < n_proc; i++) {     // 为子线程分配任务
            t[i] = std::thread(mt_relu, R + i * len_per_proc, I + i * len_per_proc, len_per_proc);
        }
        for (int i = n_proc * len_per_proc; i < len; i++) {    // 让主线程处理剩余的一点任务
            R[i] = (I[i] > 0) ? I[i] : 0;
        }
        for (int i = 0; i < n_proc; i++) {     // 等待子线程结束
            t[i].join();
        }
    }
    else {
        int new_len = len / 4 * 4;
        for(int i = 0; i<len; i+=4) {
            R[i] = (I[i] > 0) ? I[i] : 0;
            R[i+1] = (I[i+1] > 0) ? I[i+1] : 0;
            R[i+2] = (I[i+2] > 0) ? I[i+2] : 0;
            R[i+3] = (I[i+3] > 0) ? I[i+3] : 0;
        }
        for(int i = new_len; i<len; i++) {
            R[i] = (I[i] > 0) ? I[i] : 0;
        }
    }
    return result;
}

Tensor<float32> functional::padding(Tensor<float32> *input, const std::vector<int> &padding_size)
{
    /*
     * padding
     */
    int ph = padding_size[0];
    int pw = padding_size[1];
    // 计算padding后尺寸
    int batch_size = input->size[0];
    int channel = input->size[1];
    int height = input->size[2];
    int width = input->size[3];
    int padded_height = height + ph * 2;
    int padded_width = width + pw * 2;
    // 创建padding后对象
    Tensor<float32> padded{std::vector<int>{batch_size, channel, padded_height, padded_width}};
    // padding
    float32 * y_ptr = padded.data;
    float32 * x_ptr = input->data;
    for(int n = 0; n<batch_size; n++) {
        for(int c = 0; c<channel; c++) {
            // 对于前padding_height行，直接memset0
            for(int h = 0; h<ph; h++) {
                memset(y_ptr, 0, sizeof(float32)*padded_width);
                y_ptr += padded_width;
            }
            // 对于之后的xh行，先在y中memset长度为pw的0，然后memcpy x的一行，然后再mamset长度为pw的0
            for(int h = 0; h<height; h++) {
                memset(y_ptr, 0, sizeof(float32)*pw);
                y_ptr += pw;
                memcpy(y_ptr, x_ptr, sizeof(float32)*width);
                y_ptr += width;
                x_ptr += width;
                memset(y_ptr, 0, sizeof(float32)*pw);
                y_ptr += pw;
            }
            // 对于最后的ph行，直接在y中memset0
            for(int h = 0; h<ph; h++) {
                memset(y_ptr, 0, sizeof(float)*padded_width);
                y_ptr += padded_width;
            }
        }
    }
    return padded;
}
//Tensor<float32> functional::padding(Tensor<float32> *input, const std::vector<int> &padding_size) {
//    /*
//     * padding
//     */
//    // 计算padding后尺寸
//    int batch_size = input->size[0];
//    int channel = input->size[1];
//    int height = input->size[2];
//    int width = input->size[3];
//    int padded_height = height + padding_size[0] * 2;
//    int padded_width = width + padding_size[1] * 2;
//    // 创建padding后对象
//    Tensor<float32> padded{std::vector<int>{batch_size, channel, padded_height, padded_width}};
//    // padding
//    for(int n = 0; n<batch_size; n++) {
//        for(int c = 0; c<channel; c++) {
//            for(int h = 0; h<padded_height; h++) {
//                for(int w = 0; w<padded_width; w++) {
//                    if((h < padding_size[0]) || (h >= height + padding_size[0]) ||
//                       (w < padding_size[1]) || (w >= width + padding_size[1])) {
//                        // padded[n][c][h][w] = 0
//                        padded.data[
//                                n * channel * padded_height * padded_width +
//                                c * padded_height * padded_width +
//                                h * padded_width +
//                                w] = 0;
//                    }
//                    else {
//                        // padded[n][c][h][w] = input[n][c][h-ph][w-pw]
//                        padded.data[
//                                n * channel * padded_height * padded_width +
//                                c * padded_height * padded_width +
//                                h * padded_width +
//                                w]
//                                =
//                        input->data[
//                                n * channel * height * width +
//                                c * height * width +
//                                (h-padding_size[0]) * width +
//                                (w-padding_size[1])];
//                    }
//                }
//            }
//        }
//    }
//    return padded;
//}

Tensor<float32>
functional::maxpool2d(Tensor<float32> *input, const std::vector<int>& kernel_size,
                      std::vector<int> stride,
                      const std::vector<int>& padding_size,
                      const std::vector<int>& dilation) {
    /*
     * maxpool2d
     */
    // 校验参数
    if(input->size.size() != 4) {
        fprintf(stderr, "File: functional.cpp, line: %d. Dimension of input must be 4\n", __LINE__);
        exit(-1);
    }
    if(kernel_size.size() != 2) {
        fprintf(stderr, "File: functional.cpp, line: %d. Dimension of kernel_size must be 2\n", __LINE__);
        exit(-1);
    }
    if(stride.size() != 2) {
        fprintf(stderr, "File: functional.cpp, line: %d. Dimension of stride must be 2\n", __LINE__);
        exit(-1);
    }
    if(padding_size.size() != 2) {
        fprintf(stderr, "File: functional.cpp, line: %d. Dimension of padding_size must be 2\n", __LINE__);
        exit(-1);
    }
    if(dilation.size() != 2) {
        fprintf(stderr, "File: functional.cpp, line: %d. Dimension of dilation must be 2\n", __LINE__);
        exit(-1);
    }
    if(stride[0] == -1 && stride[1] == -1) {
        stride[0] = kernel_size[0];
        stride[1] = kernel_size[1];
    }
    // padding
    Tensor<float32> padded;
    if(padding_size[0] == 0 && padding_size[1] == 0) {
        padded = *input;
    }
    else {
        padded = padding(input, padding_size);
    }
    // 计算pool后尺寸
    int batch_size = input->size[0];
    int channel = input->size[1];
    int height = (padded.size[2] - (dilation[0]*(kernel_size[0]-1)+1)) / stride[0] + 1;
    int width = (padded.size[3] - (dilation[1]*(kernel_size[1]-1)+1)) / stride[1] + 1;
    // 创建返回对象
    Tensor<float32> result{std::vector<int>{batch_size, channel, height, width}};
    // pool
    for(int n = 0; n<batch_size; n++) {
        for(int c = 0; c<channel; c++) {
            for(int h = 0; h<height; h++) {
                for(int w = 0; w<width; w++) {
                    // 从padded的每块中找到最大值
                    // 1. 计算起始位置
                    int start_h = h * stride[0];    // 相对于整张图片的偏移
                    int start_w = w * stride[1];
                    int start_kh = 0;               // 相对于kernel的偏移
                    int start_kw = 0;
                    // max = padded[n][c][start_h+start_kh][start_w+start_kw]
                    float32 max = padded.data[
                            n * channel * padded.size[2] * padded.size[3] +
                            c * padded.size[2] * padded.size[3] +
                            start_h * padded.size[3] +
                            start_w];
                    for(int kh = 0; kh < kernel_size[0]; kh++, start_kh += dilation[0]) {
                        start_kw = 0;
                        for(int kw = 0; kw < kernel_size[1]; kw++, start_kw += dilation[1]) {
                            if(padded.data[
                                    n * channel * padded.size[2] * padded.size[3] +
                                    c * padded.size[2] * padded.size[3] +
                                    (start_h + start_kh) * padded.size[3] +
                                    (start_w + start_kw)] > max) {
                                max = padded.data[
                                        n * channel * padded.size[2] * padded.size[3] +
                                        c * padded.size[2] * padded.size[3] +
                                        (start_h + start_kh) * padded.size[3] +
                                        (start_w + start_kw)];
                            }
                        }
                    }
                    // result[n][c][h][w] = max
                    result.data[
                            n * channel * height * width +
                            c * height * width +
                            h * width +
                            w] = max;
                }
            }
        }
    }
    return result;
}

Tensor<float32> functional::flatten(Tensor<float32> *input) {
    /*
     * flatten
     */
    Tensor<float32> result = input->reshape(std::vector<int>{input->size[0], -1});
    return result;
}

Tensor<float32> functional::dense(Tensor<float32> *input, Tensor<float32> *weight, Tensor<float32> *bias) {
    /*
     * dense
     */
    // 检查参数
    if(input->size.size() != 2) {
        fprintf(stderr, "File functional.cpp, line %d. Only 2 dimension input is allowed in dense\n", __LINE__);
        exit(-1);
    }
    if(weight->size.size() != 2) {
        fprintf(stderr, "File functional.cpp, line %d. Only 2 dimension weight is allowed in dense\n", __LINE__);
        exit(-1);
    }
    if(bias->size.size() != 1) {
        fprintf(stderr, "File functional.cpp, line %d. Only 1 dimension bias is allowed in dense\n", __LINE__);
        exit(-1);
    }
    // 矩阵乘法
    Tensor<float32> dot_res = (*input).dot(*weight);
    // +bias
    for(int n = 0; n<input->size[0]; n++) {
        for(int l = 0; l<weight->size[1]; l++) {
            dot_res.data[n * weight->size[1] + l] += bias->data[l];
        }
    }
    return dot_res;
}

Tensor<float32> functional::add(Tensor<float32> *input1, Tensor<float32> *input2) {
    /*
     * add
     */
    return (*input1) + (*input2);
}

Tensor<float32> functional::concat(Tensor<float32> *input1, Tensor<float32> *input2, int dim) {
    /*
     * concat
     */
    return input1->concat(*input2, dim);
}

Tensor<float32>
functional::batch_norm2d(Tensor<float32> *input, Tensor<float32> *running_mean, Tensor<float32> *running_var,
                       Tensor<float32> *weight, Tensor<float32> *bias, float eps)
{
    /*
     * batch normalization 2d
     */
    if(input->size.size() != 4) {
        fprintf(stderr, "File functional.cpp, line %d. Only 4 dimension input is allowed in batch_norm2d\n", __LINE__);
        exit(-1);
    }
    if(running_mean->size.size() != 1) {
        fprintf(stderr, "File functional.cpp, line %d. Only 1 dimension running_mean is allowed in batch_norm2d\n", __LINE__);
        exit(-1);
    }
    if(running_var->size.size() != 1) {
        fprintf(stderr, "File functional.cpp, line %d. Only 1 dimension running_var is allowed in batch_norm2d\n", __LINE__);
        exit(-1);
    }
    if(weight->size.size() != 1) {
        fprintf(stderr, "File functional.cpp, line %d. Only 1 dimension weight is allowed in batch_norm2d\n", __LINE__);
        exit(-1);
    }
    if(bias->size.size() != 1) {
        fprintf(stderr, "File functional.cpp, line %d. Only 1 dimension bias is allowed in batch_norm2d\n", __LINE__);
        exit(-1);
    }
    Tensor<float32> x = input->deep_copy();
    Tensor<float32> E_x = running_mean->deep_copy();
    Tensor<float32> Var_x = running_var->deep_copy();
    int batch_size = x.size[0];
    int channel = x.size[1];
    for(int n = 0; n < batch_size; n++) {
        for (int c = 0; c < channel; c++) {
            x[n][c] -= E_x[c].to_num();
        }
    }
    Var_x += eps;
    Var_x = Var_x.elewise_sqrt();
    Tensor<float32> y = (x / Var_x.reshape(std::vector<int>{1,-1,1,1}))
            * (weight->reshape(std::vector<int>{1,-1,1,1}))
            + (bias->reshape(std::vector<int>{1,-1,1,1}));
    return y;
}

Tensor<float32> functional::avgpool2d(Tensor<float32> *input, const std::vector<int> &kernel_size,
                                      std::vector<int> stride, const std::vector<int> &padding_size) {
    /*
     * avgpool2d
     */
    // 校验参数
    if(input->size.size() != 4) {
        fprintf(stderr, "File: functional.cpp, line: %d. Dimension of input must be 4\n", __LINE__);
        exit(-1);
    }
    if(kernel_size.size() != 2) {
        fprintf(stderr, "File: functional.cpp, line: %d. Dimension of kernel_size must be 2\n", __LINE__);
        exit(-1);
    }
    if(stride.size() != 2) {
        fprintf(stderr, "File: functional.cpp, line: %d. Dimension of stride must be 2\n", __LINE__);
        exit(-1);
    }
    if(padding_size.size() != 2) {
        fprintf(stderr, "File: functional.cpp, line: %d. Dimension of padding_size must be 2\n", __LINE__);
        exit(-1);
    }
    if(stride[0] == -1 && stride[1] == -1) {
        stride[0] = kernel_size[0];
        stride[1] = kernel_size[1];
    }
    // padding
    Tensor<float32> padded;
    if(padding_size[0] == 0 && padding_size[1] == 0) {
        padded = *input;
    }
    else {
        padded = padding(input, padding_size);
    }
    // 计算pool后尺寸
    int batch_size = input->size[0];
    int channel = input->size[1];
    int height = (padded.size[2] - kernel_size[0]) / stride[0] + 1;
    int width = (padded.size[3] - kernel_size[1]) / stride[1] + 1;
    // 创建返回对象
    Tensor<float32> result{std::vector<int>{batch_size, channel, height, width}};
    // pool
    for(int n = 0; n<batch_size; n++) {
        for(int c = 0; c<channel; c++) {
            for(int h = 0; h<height; h++) {
                for(int w = 0; w<width; w++) {
                    // 从padded的每块中找到最大值
                    // 1. 计算起始位置
                    int start_h = h * stride[0];    // 相对于整张图片的偏移
                    int start_w = w * stride[1];
                    int start_kh = 0;               // 相对于kernel的偏移
                    int start_kw = 0;
                    // item = padded[n][c][start_h+start_kh][start_w+start_kw]
                    float32 sum = 0;
                    for(int kh = 0; kh < kernel_size[0]; kh++, start_kh ++) {
                        start_kw = 0;
                        for(int kw = 0; kw < kernel_size[1]; kw++, start_kw ++) {
                            sum += padded.data[
                                        n * channel * padded.size[2] * padded.size[3] +
                                        c * padded.size[2] * padded.size[3] +
                                        (start_h + start_kh) * padded.size[3] +
                                        (start_w + start_kw)];
                        }
                    }
                    // result[n][c][h][w] = sum / (kernel_size[0] * kernel_size[1])
                    result.data[
                            n * channel * height * width +
                            c * height * width +
                            h * width +
                            w] = sum / (float32)(kernel_size[0] * kernel_size[1]);
                }
            }
        }
    }
    return result;
}

Tensor<float32> 
functional::dropout(Tensor<float32> *input, const float p)
{
    /*
     * dropout
     */
    Tensor<float32> ret;
    ret = (*input) * (1-p);
    return ret;
}

// Tensor<uint8>
// functional::qconv2d(Tensor<uint8> *input, int zero_x, int zero_w, int zero_b, int zero_y,
//                     Fixed_point coe, int rshift, int qmin, int qmax,
//                     Tensor<uint8> *weight, Tensor<int32> *bias, const std::vector<int> &stride,
//                     const std::vector<int> &padding_size, const std::vector<int> &dilation) {
//     /*
//      * qconv2d
//      */
//     // 对输入尺寸进行校验
//     if(input->size.size() != 4) {
//         fprintf(stderr, "only 4 dimension input is allowed in conv2d\n");
//         exit(-1);
//     }
//     if(weight->size.size() != 4) {
//         fprintf(stderr, "only 4 dimension weight is allowed in conv2d\n");
//         exit(-1);
//     }
//     if(input->size[1] != weight->size[1]) {
//         fprintf(stderr, "channel of input should equal to input channel of weight\n");
//         exit(-1);
//     }
//     if(bias->size.size() != 1) {
//         fprintf(stderr, "only 1 dimension bias is allowed in conv2d\n");
//         exit(-1);
//     }
//     if(bias->size[0] != weight->size[0]) {
//         fprintf(stderr, "dim 1 of bias should equal to dim 1 of weight\n");
//     }
//     if(stride.size() != 2) {
//         fprintf(stderr, "stride should be a vector of 2 int\n");
//         exit(-1);
//     }
//     if(padding_size.size() != 2) {
//         fprintf(stderr, "padding should be a vector of 2 int\n");
//         exit(-1);
//     }
//     if(dilation.size() != 2) {
//         fprintf(stderr, "dilation should be a vector of 2 int\n");
//         exit(-1);
//     }
//     // padding
//     Tensor<uint8> padded = qpadding(input, padding_size, zero_x);
//     // qconv2d
//     // 计算输出尺寸以及其他尺寸
//     std::vector<int> kernel_size{weight->size[2], weight->size[3]};
//     int batch_size = padded.size[0];
//     int output_channel = weight->size[0];
//     int height = (padded.size[2] - (dilation[0] * (kernel_size[0]-1) + 1)) / stride[0] + 1;
//     int width = (padded.size[3] - (dilation[1] * (kernel_size[1]-1) + 1)) / stride[1] + 1;
//     int input_channel = input->size[1];
//     int kernel_height = kernel_size[0];
//     int kernel_width = kernel_size[1];
//     // 创建中间结果
//     Tensor<int32> result{std::vector<int>{batch_size, output_channel, height, width}};
//     // 计算
//     Fixed_point fp_temp{0};
//     for(int n = 0; n<batch_size; n++) {
//         // 每次处理一张图片
//         for(int o = 0; o<output_channel; o++) {
//             int start_h = 0;
//             for(int h = 0; h<height; h++, start_h += stride[0]) {
//                 int start_w = 0;
//                 for(int w = 0; w<width; w++, start_w += stride[1]) {
//                     int temp = 0;
//                     // temp += input_channel * kernel_height * kernel_width * zero_x * zero_w;
//                     for(int i = 0; i<input_channel; i++) {
//                         for(int kh = 0; kh < kernel_height; kh++) {
//                             for(int kw = 0; kw < kernel_width; kw++) {
// //                                temp += padded[n][i][h+kh*dilation[0]][w+kw*dilation[1]] * weight[o][i][kh][kw];
//                                 temp += (int)(padded.data[
//                                         n * padded.size[1] * padded.size[2] * padded.size[3] +
//                                         i * padded.size[2] * padded.size[3] +
//                                         (start_h+kh*dilation[0]) * padded.size[3] +
//                                         (start_w+kw*dilation[1])] - zero_x)
//                                                 *
//                                         (int)(weight->data[
//                                         o * weight->size[1] * weight->size[2] * weight->size[3] +
//                                         i * weight->size[2] * weight->size[3] +
//                                         kh * weight->size[3] +
//                                         kw] - zero_w);
//                             }
//                         }
//                     }
// //                     for(int i = 0; i<input_channel; i++) {
// //                         for(int kh = 0; kh < kernel_height; kh++) {
// //                             for(int kw = 0; kw < kernel_width; kw++) {
// // //                                temp -= zero_x * weight[o][i][kh][kw];
// // //                                temp -= zero_w * padded[n][i][h+kh*dilation[0]][w+kw*dilation[1]];
// //                                 temp -= zero_x * weight->data[
// //                                         o * weight->size[1] * weight->size[2] * weight->size[3] +
// //                                         i * weight->size[2] * weight->size[3] +
// //                                         kh * weight->size[3] +
// //                                         kw];
// //                                 temp -= zero_w * padded.data[
// //                                         n * padded.size[1] * padded.size[2] * padded.size[3] +
// //                                         i * padded.size[2] * padded.size[3] +
// //                                         (start_h+kh*dilation[0]) * padded.size[3] +
// //                                         (start_w+kw*dilation[1])];
// //                             }
// //                         }
// //                     }
//                     temp += bias->data[o] - zero_b;
//                     fp_temp.assign(temp);
//                     fp_temp *= coe;
//                     int t = fp_temp.to_int();
//                     result.data[
//                             n * result.size[1] * result.size[2] * result.size[3] +
//                             o * result.size[2] * result.size[3] +
//                             h * result.size[3] +
//                             w] = (t >> rshift) + zero_y;
//                 }
//             }
//         }
//     }
//     // 将中间结果clip后转为uint8并返回
//     result.clip(qmin, qmax);
//     Tensor<uint8> ret = result.astype_uint8();
//     return ret;
// }

void qconv2d_thread(int n, int start_o, int end_o, int height, int width, 
                    std::vector<int> stride, std::vector<int> dilation, 
                    int input_channel, int kernel_height, int kernel_width,
                    int zero_x, int zero_w, int zero_b, int zero_y,
                    uint8 * padded_data, std::vector<int> padded_size, 
                    int32 * result_data, std::vector<int> result_size,
                    uint8 * weight_data, std::vector<int> weight_size,
                    int32 * bias_data,  
                    Fixed_point coe, int rshift
                    )
{
    Fixed_point fp_temp{0};
    for(int o = start_o; o<end_o; o++) {
        int start_h = 0;
        for(int h = 0; h<height; h++, start_h += stride[0]) {
            int start_w = 0;
            for(int w = 0; w<width; w++, start_w += stride[1]) {
                int temp = 0;
                temp += input_channel * kernel_height * kernel_width * zero_x * zero_w;
                for(int i = 0; i<input_channel; i++) {
                    for(int kh = 0; kh < kernel_height; kh++) {
                        for(int kw = 0; kw < kernel_width; kw++) {
    //                                temp += padded[n][i][h+kh*dilation[0]][w+kw*dilation[1]] * weight[o][i][kh][kw];
                            temp += padded_data[
                                    n * padded_size[1] * padded_size[2] * padded_size[3] +
                                    i * padded_size[2] * padded_size[3] +
                                    (start_h+kh*dilation[0]) * padded_size[3] +
                                    (start_w+kw*dilation[1])]
                                            *
                                    weight_data[
                                    o * weight_size[1] * weight_size[2] * weight_size[3] +
                                    i * weight_size[2] * weight_size[3] +
                                    kh * weight_size[3] +
                                    kw];
                        }
                    }
                }
                for(int i = 0; i<input_channel; i++) {
                    for(int kh = 0; kh < kernel_height; kh++) {
                        for(int kw = 0; kw < kernel_width; kw++) {
    //                                temp -= zero_x * weight[o][i][kh][kw];
    //                                temp -= zero_w * padded[n][i][h+kh*dilation[0]][w+kw*dilation[1]];
                            temp -= zero_x * weight_data[
                                    o * weight_size[1] * weight_size[2] * weight_size[3] +
                                    i * weight_size[2] * weight_size[3] +
                                    kh * weight_size[3] +
                                    kw];
                            temp -= zero_w * padded_data[
                                    n * padded_size[1] * padded_size[2] * padded_size[3] +
                                    i * padded_size[2] * padded_size[3] +
                                    (start_h+kh*dilation[0]) * padded_size[3] +
                                    (start_w+kw*dilation[1])];
                        }
                    }
                }
                temp += bias_data[o] - zero_b;
                fp_temp.assign(temp);
                fp_temp *= coe;
                int t = fp_temp.to_int();
                result_data[
                        n * result_size[1] * result_size[2] * result_size[3] +
                        o * result_size[2] * result_size[3] +
                        h * result_size[3] +
                        w] = (t >> rshift) + zero_y;
            }
        }
    }
}

Tensor<uint8>
functional::qconv2d(Tensor<uint8> *input, int zero_x, int zero_w, int zero_b, int zero_y,
                    Fixed_point coe, int rshift, int qmin, int qmax,
                    Tensor<uint8> *weight, Tensor<int32> *bias, const std::vector<int> &stride,
                    const std::vector<int> &padding_size, const std::vector<int> &dilation) {
    /*
     * qconv2d
     */
    // 对输入尺寸进行校验
    if(input->size.size() != 4) {
        fprintf(stderr, "only 4 dimension input is allowed in conv2d\n");
        exit(-1);
    }
    if(weight->size.size() != 4) {
        fprintf(stderr, "only 4 dimension weight is allowed in conv2d\n");
        exit(-1);
    }
    if(input->size[1] != weight->size[1]) {
        fprintf(stderr, "channel of input should equal to input channel of weight\n");
        exit(-1);
    }
    if(bias->size.size() != 1) {
        fprintf(stderr, "only 1 dimension bias is allowed in conv2d\n");
        exit(-1);
    }
    if(bias->size[0] != weight->size[0]) {
        fprintf(stderr, "dim 1 of bias should equal to dim 1 of weight\n");
    }
    if(stride.size() != 2) {
        fprintf(stderr, "stride should be a vector of 2 int\n");
        exit(-1);
    }
    if(padding_size.size() != 2) {
        fprintf(stderr, "padding should be a vector of 2 int\n");
        exit(-1);
    }
    if(dilation.size() != 2) {
        fprintf(stderr, "dilation should be a vector of 2 int\n");
        exit(-1);
    }
    // padding
    Tensor<uint8> padded = qpadding(input, padding_size, zero_x);
    // qconv2d
    // 计算输出尺寸以及其他尺寸
    std::vector<int> kernel_size{weight->size[2], weight->size[3]};
    int batch_size = padded.size[0];
    int output_channel = weight->size[0];
    int height = (padded.size[2] - (dilation[0] * (kernel_size[0]-1) + 1)) / stride[0] + 1;
    int width = (padded.size[3] - (dilation[1] * (kernel_size[1]-1) + 1)) / stride[1] + 1;
    int input_channel = input->size[1];
    int kernel_height = kernel_size[0];
    int kernel_width = kernel_size[1];
    // 创建中间结果
    Tensor<int32> result{std::vector<int>{batch_size, output_channel, height, width}};
    // 计算
    for(int n = 0; n<batch_size; n++) {
        // 每次处理一张图片
        int n_proc = sys_info->n_proc;
        int channel_per_thread = output_channel / n_proc;   // 分给每个线程的通道数
        int channel_left = output_channel % n_proc;         // 剩下的通道数，分给最后一个线程
        std::thread t[n_proc];
        for(int i = 0; i<n_proc; i++) {
            int start_o = i * channel_per_thread;
            int end_o = start_o + channel_per_thread;
            if(i == n_proc-1) {
                end_o += channel_left;
            }
            t[i] = std::thread(
                qconv2d_thread, n, start_o, end_o, height, width, 
                stride, dilation, input_channel, kernel_height, kernel_width, 
                zero_x, zero_w, zero_b, zero_y, 
                padded.data, padded.size, 
                result.data, result.size,
                weight->data, weight->size, 
                bias->data, coe, rshift);
        }
        for(int i = 0; i<n_proc; i++) {
            t[i].join();
        }
    }
    // 将中间结果clip后转为uint8并返回
    result.clip(qmin, qmax);
    Tensor<uint8> ret = result.astype_uint8();
    return ret;
}

Tensor<uint8> functional::qpadding(Tensor<uint8> *input, const std::vector<int> &padding_size, int zero)
{
    /*
     * qpadding
     */
    int ph = padding_size[0];
    int pw = padding_size[1];
    // 计算padding后尺寸
    int batch_size = input->size[0];
    int channel = input->size[1];
    int height = input->size[2];
    int width = input->size[3];
    int padded_height = height + ph * 2;
    int padded_width = width + pw * 2;
    // 创建padding后对象
    Tensor<uint8> padded{std::vector<int>{batch_size, channel, padded_height, padded_width}};
    // padding
    uint8 * y_ptr = padded.data;
    uint8 * x_ptr = input->data;
    for(int n = 0; n<batch_size; n++) {
        for(int c = 0; c<channel; c++) {
            // 对于前ph行，直接在y中memset zero
            for(int h = 0; h<ph; h++) {
                memset(y_ptr, zero, sizeof(uint8)*padded_width);
                y_ptr += padded_width;
            }
            // 对于之后的xh行，先在y中memset长度为pw的0，然后memcpy x的一行，然后再mamset长度为pw的0
            for(int h = 0; h<height; h++) {
                memset(y_ptr, zero, sizeof(uint8)*pw);
                y_ptr += pw;
                memcpy(y_ptr, x_ptr, sizeof(uint8)*width);
                y_ptr += width;
                x_ptr += width;
                memset(y_ptr, zero, sizeof(uint8)*pw);
                y_ptr += pw;
            }
            // 对于最后的ph行，直接在y中memset zero
            for(int h = 0; h<ph; h++) {
                memset(y_ptr, zero, sizeof(uint8)*padded_width);
                y_ptr += padded_width;
            }
        }
    }
    return padded;
}
//Tensor<uint8> functional::qpadding(Tensor<uint8> *input, const std::vector<int> &padding_size, int zero) {
//    /*
//     * qpadding
//     */
//    // 计算padding后尺寸
//    int batch_size = input->size[0];
//    int channel = input->size[1];
//    int height = input->size[2];
//    int width = input->size[3];
//    int padded_height = height + padding_size[0] * 2;
//    int padded_width = width + padding_size[1] * 2;
//    // 创建padding后对象
//    Tensor<uint8> padded{std::vector<int>{batch_size, channel, padded_height, padded_width}};
//    // padding
//    for(int n = 0; n<batch_size; n++) {
//        for(int c = 0; c<channel; c++) {
//            for(int h = 0; h<padded_height; h++) {
//                for(int w = 0; w<padded_width; w++) {
//                    if((h < padding_size[0]) || (h >= height + padding_size[0]) ||
//                       (w < padding_size[1]) || (w >= width + padding_size[1])) {
//                        // padded[n][c][h][w] = 0
//                        padded.data[
//                                n * channel * padded_height * padded_width +
//                                c * padded_height * padded_width +
//                                h * padded_width +
//                                w] = zero;
//                    }
//                    else {
//                        // padded[n][c][h][w] = input[n][c][h-ph][w-pw]
//                        padded.data[
//                                n * channel * padded_height * padded_width +
//                                c * padded_height * padded_width +
//                                h * padded_width +
//                                w]
//                                =
//                                input->data[
//                                        n * channel * height * width +
//                                        c * height * width +
//                                        (h-padding_size[0]) * width +
//                                        (w-padding_size[1])];
//                    }
//                }
//            }
//        }
//    }
//    return padded;
//}

Tensor<uint8> functional::qrelu(Tensor<uint8> *input, int zero, int qmax) {
    /*
     * qrelu
     */
    Tensor<uint8> res{input->size};
    int len = res.len();
    for(int i = 0; i<len; i++) {
        res.data[i] = clip(input->data[i], zero, qmax);
    }
    return res;
}

Tensor<uint8> functional::qmaxpool2d(Tensor<uint8> *input, int zero, const std::vector<int> &kernel_size,
                                     std::vector<int> stride, const std::vector<int> &padding_size,
                                     const std::vector<int> &dilation)
{
    /*
     * qmaxpool2d
     */
    // 校验参数
    if(input->size.size() != 4) {
        fprintf(stderr, "File: functional.cpp, line: %d. Dimension of input must be 4\n", __LINE__);
        exit(-1);
    }
    if(kernel_size.size() != 2) {
        fprintf(stderr, "File: functional.cpp, line: %d. Dimension of kernel_size must be 2\n", __LINE__);
        exit(-1);
    }
    if(stride.size() != 2) {
        fprintf(stderr, "File: functional.cpp, line: %d. Dimension of stride must be 2\n", __LINE__);
        exit(-1);
    }
    if(padding_size.size() != 2) {
        fprintf(stderr, "File: functional.cpp, line: %d. Dimension of padding_size must be 2\n", __LINE__);
        exit(-1);
    }
    if(dilation.size() != 2) {
        fprintf(stderr, "File: functional.cpp, line: %d. Dimension of dilation must be 2\n", __LINE__);
        exit(-1);
    }
    if(stride[0] == -1 && stride[1] == -1) {
        stride[0] = kernel_size[0];
        stride[1] = kernel_size[1];
    }
    // padding
    Tensor<uint8> padded;
    if(padding_size[0] == 0 && padding_size[1] == 0) {
        padded = *input;
    }
    else {
        padded = qpadding(input, padding_size, zero);
    }
    // 计算pool后尺寸
    int batch_size = input->size[0];
    int channel = input->size[1];
    int height = (padded.size[2] - (dilation[0]*(kernel_size[0]-1)+1)) / stride[0] + 1;
    int width = (padded.size[3] - (dilation[1]*(kernel_size[1]-1)+1)) / stride[1] + 1;
    // 创建返回对象
    Tensor<uint8> result{std::vector<int>{batch_size, channel, height, width}};
    // pool
    for(int n = 0; n<batch_size; n++) {
        for(int c = 0; c<channel; c++) {
            for(int h = 0; h<height; h++) {
                for(int w = 0; w<width; w++) {
                    // 从padded的每块中找到最大值
                    // 1. 计算起始位置
                    int start_h = h * stride[0];    // 相对于整张图片的偏移
                    int start_w = w * stride[1];
                    int start_kh = 0;               // 相对于kernel的偏移
                    int start_kw = 0;
                    // max = padded[n][c][start_h+start_kh][start_w+start_kw]
                    uint8 max = padded.data[
                            n * channel * padded.size[2] * padded.size[3] +
                            c * padded.size[2] * padded.size[3] +
                            start_h * padded.size[3] +
                            start_w];
                    for(int kh = 0; kh < kernel_size[0]; kh++, start_kh += dilation[0]) {
                        start_kw = 0;
                        for(int kw = 0; kw < kernel_size[1]; kw++, start_kw += dilation[1]) {
                            if(padded.data[
                                       n * channel * padded.size[2] * padded.size[3] +
                                       c * padded.size[2] * padded.size[3] +
                                       (start_h + start_kh) * padded.size[3] +
                                       (start_w + start_kw)] > max) {
                                max = padded.data[
                                        n * channel * padded.size[2] * padded.size[3] +
                                        c * padded.size[2] * padded.size[3] +
                                        (start_h + start_kh) * padded.size[3] +
                                        (start_w + start_kw)];
                            }
                        }
                    }
                    // result[n][c][h][w] = max
                    result.data[
                            n * channel * height * width +
                            c * height * width +
                            h * width +
                            w] = max;
                }
            }
        }
    }
    return result;
}

Tensor<uint8> functional::qflatten(Tensor<uint8> *input) {
    /*
     * qflatten
     */
    Tensor<uint8> result = input->reshape(std::vector<int>{input->size[0], -1});
    return result;
}

Tensor<uint8>
functional::qdense(Tensor<uint8> *input, int zero_x, int zero_w, int zero_b, int zero_y, Fixed_point coe, int rshift,
                   int qmin, int qmax, Tensor<uint8> *weight, Tensor<int32> *bias) {
    /*
     * qdense
     */
    // 检查参数
    if(input->size.size() != 2) {
        fprintf(stderr, "File functional.cpp, line %d. Only 2 dimension input is allowed in dense\n", __LINE__);
        exit(-1);
    }
    if(weight->size.size() != 2) {
        fprintf(stderr, "File functional.cpp, line %d. Only 2 dimension weight is allowed in dense\n", __LINE__);
        exit(-1);
    }
    if(bias->size.size() != 1) {
        fprintf(stderr, "File functional.cpp, line %d. Only 1 dimension bias is allowed in dense\n", __LINE__);
        exit(-1);
    }
    // 矩阵乘法
    Tensor<int> result{std::vector<int>{input->size[0], weight->size[1]}};
    int batch_size = input->size[0];
    int output_channel = weight->size[1];
    int input_channel = input->size[1];
    Fixed_point fp_temp{0};
    for(int n = 0; n<batch_size; n++) {
        for (int o = 0; o < output_channel; o++) {
            int temp = 0;
            temp += input_channel * zero_x * zero_w;
            for(int i = 0; i<input_channel; i++) {
                temp += input->data[n * input_channel + i] * weight->data[i * output_channel + o];
            }
            for(int i = 0; i<input_channel; i++) {
                temp -= zero_x * weight->data[i * output_channel + o];
                temp -= zero_w * input->data[n * input_channel + i];
            }
            temp += bias->data[o] - zero_b;
            fp_temp.assign(temp);
            fp_temp *= coe;
            int t = fp_temp.to_int();
            result.data[n * output_channel + o] = (t >> rshift) + zero_y;
        }
    }
    result.clip(qmin, qmax);
    Tensor<uint8> ret = result.astype_uint8();
    return ret;
}

// Tensor<uint8>
// functional::qadd(Tensor<uint8> *input1, Tensor<uint8> *input2, int zero_x1, int zero_x2, int zero_y,
//                  Fixed_point coe1, Fixed_point coe2, int rshift1, int rshift2, int qmin, int qmax) {
//     /*
//      * qadd
//      */
//     Tensor<int32> temp_x1 = input1->astype_int32();
//     Tensor<int32> temp_x2 = input2->astype_int32();
//     Fixed_point fp_temp1{0};
//     Fixed_point fp_temp2{0};
//     int len1 = input1->len();
//     for(int i = 0; i<len1; i++) {
//         int temp1 = temp_x1.data[i] - zero_x1;
//         fp_temp1.assign(temp1);
//         fp_temp1 *= coe1;
//         int t1 = fp_temp1.to_int();
//         temp_x1.data[i] = t1 >> rshift1;
//     }
//     int len2 = input2->len();
//     for(int i = 0; i<len2; i++) {
//         int temp2 = temp_x2.data[i] - zero_x2;
//         fp_temp2.assign(temp2);
//         fp_temp2 *= coe2;
//         int t2 = fp_temp2.to_int();
//         temp_x2.data[i] = t2 >> rshift2;
//     }
//     Tensor<int32> result = temp_x1 + temp_x2 + zero_y;
//     result.clip(qmin, qmax);
//     Tensor<uint8> ret = result.astype_uint8();
//     return ret;
// }

Tensor<uint8>
functional::qadd(Tensor<uint8> *input1, Tensor<uint8> *input2, int zero_x1, int zero_x2, int zero_y,
                 Fixed_point coe1, Fixed_point coe2, int rshift1, int rshift2, int qmin, int qmax) {
    /*
     * qadd
     */
    Tensor<int32> temp_x1 = input1->astype_int32();
    Tensor<int32> temp_x2 = input2->astype_int32();
    Fixed_point fp_temp1{0};
    Fixed_point fp_temp2{0};
    int len1 = input1->len();
    for(int i = 0; i<len1; i++) {
        int temp1 = temp_x1.data[i] - zero_x1;
        fp_temp1.assign(temp1);
        fp_temp1 *= coe1;
        int t1 = fp_temp1.to_int();
        if(rshift1 < 0) {
            temp_x1.data[i] = t1 << (-rshift1);
        }
        else {
            temp_x1.data[i] = t1 >> rshift1;
        }
    }
    int len2 = input2->len();
    for(int i = 0; i<len2; i++) {
        int temp2 = temp_x2.data[i] - zero_x2;
        fp_temp2.assign(temp2);
        fp_temp2 *= coe2;
        int t2 = fp_temp2.to_int();
        if(rshift2 < 0) {
            temp_x2.data[i] = t2 << (-rshift2);
        }
        else {
            temp_x2.data[i] = t2 >> rshift2;
        }
    }
    Tensor<int32> result = temp_x1 + temp_x2 + zero_y;
    result.clip(qmin, qmax);
    Tensor<uint8> ret = result.astype_uint8();
    return ret;
}

Tensor<uint8> functional::qconcat(Tensor<uint8> *input1, Tensor<uint8> *input2, int zero_x1, int zero_x2,
                                  int zero_y, Fixed_point coe1, Fixed_point coe2, int rshift1, int rshift2,
                                  int qmin, int qmax, int dim)
{
    /*
     * qconcat
     */
    Tensor<int32> temp_x1 = input1->astype_int32();
    Tensor<int32> temp_x2 = input2->astype_int32();
    Fixed_point fp_temp1{0};
    Fixed_point fp_temp2{0};
    int len1 = input1->len();
    for(int i = 0; i<len1; i++) {
        int temp1 = temp_x1.data[i] - zero_x1;
        fp_temp1.assign(temp1);
        fp_temp1 *= coe1;
        int t1 = fp_temp1.to_int();
        temp_x1.data[i] = (t1 >> rshift1) + zero_y;
    }
    int len2 = input2->len();
    for(int i = 0; i<len2; i++) {
        int temp2 = temp_x2.data[i] - zero_x2;
        fp_temp2.assign(temp2);
        fp_temp2 *= coe2;
        int t2 = fp_temp2.to_int();
        temp_x2.data[i] = (t2 >> rshift2) + zero_y;
    }
    Tensor<int32> result = temp_x1.concat(temp_x2, dim);
    result.clip(qmin, qmax);
    Tensor<uint8> ret = result.astype_uint8();
    return ret;
}

Tensor<uint8> functional::qavgpool2d(Tensor<uint8> *input, int zero, const std::vector<int> &kernel_size,
                                     std::vector<int> stride, const std::vector<int> &padding_size) {
    /*
     * qavgpool2d
     */
    // 校验参数
    if(input->size.size() != 4) {
        fprintf(stderr, "File: functional.cpp, line: %d. Dimension of input must be 4\n", __LINE__);
        exit(-1);
    }
    if(kernel_size.size() != 2) {
        fprintf(stderr, "File: functional.cpp, line: %d. Dimension of kernel_size must be 2\n", __LINE__);
        exit(-1);
    }
    if(stride.size() != 2) {
        fprintf(stderr, "File: functional.cpp, line: %d. Dimension of stride must be 2\n", __LINE__);
        exit(-1);
    }
    if(padding_size.size() != 2) {
        fprintf(stderr, "File: functional.cpp, line: %d. Dimension of padding_size must be 2\n", __LINE__);
        exit(-1);
    }
    if(stride[0] == -1 && stride[1] == -1) {
        stride[0] = kernel_size[0];
        stride[1] = kernel_size[1];
    }
    // padding
    Tensor<uint8> padded;
    if(padding_size[0] == 0 && padding_size[1] == 0) {
        padded = *input;
    }
    else {
        padded = qpadding(input, padding_size, zero);
    }
    // 计算pool后尺寸
    int batch_size = input->size[0];
    int channel = input->size[1];
    int height = (padded.size[2] - kernel_size[0]) / stride[0] + 1;
    int width = (padded.size[3] - kernel_size[1]) / stride[1] + 1;
    // 创建返回对象
    Tensor<uint8> result{std::vector<int>{batch_size, channel, height, width}};
    // pool
    for(int n = 0; n<batch_size; n++) {
        for(int c = 0; c<channel; c++) {
            for(int h = 0; h<height; h++) {
                for(int w = 0; w<width; w++) {
                    // 从padded的每块中找到最大值
                    // 1. 计算起始位置
                    int start_h = h * stride[0];    // 相对于整张图片的偏移
                    int start_w = w * stride[1];
                    int start_kh = 0;               // 相对于kernel的偏移
                    int start_kw = 0;
                    // item = padded[n][c][start_h+start_kh][start_w+start_kw]
                    int sum = 0;
                    for(int kh = 0; kh < kernel_size[0]; kh++, start_kh ++) {
                        start_kw = 0;
                        for(int kw = 0; kw < kernel_size[1]; kw++, start_kw ++) {
                            sum += padded.data[
                                        n * channel * padded.size[2] * padded.size[3] +
                                        c * padded.size[2] * padded.size[3] +
                                        (start_h + start_kh) * padded.size[3] +
                                        (start_w + start_kw)];
                        }
                    }
                    // result[n][c][h][w] = sum / (kernel_size[0] * kernel_size[1])
                    result.data[
                            n * channel * height * width +
                            c * height * width +
                            h * width +
                            w] = (uint8)(sum / (kernel_size[0] * kernel_size[1]));
                }
            }
        }
    }
    return result;
}
