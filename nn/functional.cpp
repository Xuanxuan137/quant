//
// Created by noname on 2021/11/3.
//

#include "functional.h"

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
        Tensor<float32> result_matrix = weight_matrix.dot(input_matrix);
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
//    for(int i = 0; i<len; i++) {
//        R[i] = (I[i] > 0) ? I[i] : 0;
//    }
//    return result;
//}

Tensor<float32>
functional::relu(Tensor<float32> *input) {
    /*
     * 优化Relu
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
        mt_relu(R, I, len);
    }
    return result;
}

Tensor<float32> functional::padding(Tensor<float32> *input, const std::vector<int> &padding_size) {
    /*
     * padding
     */
    // 计算padding后尺寸
    int batch_size = input->size[0];
    int channel = input->size[1];
    int height = input->size[2];
    int width = input->size[3];
    int padded_height = height + padding_size[0] * 2;
    int padded_width = width + padding_size[1] * 2;
    // 创建padding后对象
    Tensor<float32> padded{std::vector<int>{batch_size, channel, padded_height, padded_width}};
    // padding
    for(int n = 0; n<batch_size; n++) {
        for(int c = 0; c<channel; c++) {
            for(int h = 0; h<padded_height; h++) {
                for(int w = 0; w<padded_width; w++) {
                    if((h < padding_size[0]) || (h >= height + padding_size[0]) ||
                       (w < padding_size[1]) || (w >= width + padding_size[1])) {
                        // padded[n][c][h][w] = 0
                        padded.data[
                                n * channel * padded_height * padded_width +
                                c * padded_height * padded_width +
                                h * padded_width +
                                w] = 0;
                    }
                    else {
                        // padded[n][c][h][w] = input[n][c][h-ph][w-pw]
                        padded.data[
                                n * channel * padded_height * padded_width +
                                c * padded_height * padded_width +
                                h * padded_width +
                                w]
                                =
                        input->data[
                                n * channel * height * width +
                                c * height * width +
                                (h-padding_size[0]) * width +
                                (w-padding_size[1])];
                    }
                }
            }
        }
    }
    return padded;
}

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
