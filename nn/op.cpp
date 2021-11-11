//
// Created by noname on 2021/10/29.
//

#include "op.h"

Input::Input(const std::vector <std::string>& parameters)
{
    /*
     * 分析参数为算子设置属性。可能的参数对包括：
     * shape=(1,1,28,28)
     * dtype=float32
     */
    // 遍历参数对
    for(const std::string &s: parameters) {
        // 切分参数对，分为参数和参数值
        std::vector<std::string> para_pair = split(s, "=");
        // 不同参数进行不同处理
        if(para_pair[0] == "shape") {   // shape=(1,1,28,28)
            std::string temp = replace(para_pair[1], "(", "");
            temp = replace(temp, ")", "");
            std::vector<std::string> shape_str = split(temp, ",");
            for(const std::string &shape: shape_str) {
                output_shape.push_back((int)strtol(shape.c_str(), nullptr, 10));
            }
        }
        else if(para_pair[0] == "dtype") {  // dtype=float32
            // do nothing.
            // 在最新设计中，数据类型和算子名称绑定，此参数无效
        }
    }
}

void Input::forward(Tensor<float32> *input, Tensor<float32> *output) {
    /*
     * Input算子的forward
     */
    *output = *input;
}

Input::~Input() = default;

Relu::Relu(const std::vector<std::string>& parameters,
           const std::vector<std::vector<int> > &output_shape_list)
{
    /*
     * 分析参数为算子设置属性。可能的参数对包括：
     * input=%1
     */
    // 遍历参数对
    for(const std::string &s: parameters) {
        std::vector<std::string> para_pair = split(s, "=");
        if(para_pair[0] == "input") {
            std::string temp = replace(para_pair[1], "%", "");
            this->input_node = (int)strtol(temp.c_str(), nullptr, 10);
        }
    }
    // 设置output_shape
    output_shape = output_shape_list[input_node];
}

void Relu::forward(Tensor<float32> *input, Tensor<float32> *output) {
    /*
     * Relu算子的forward
     */
    *output = F::relu(input);
}

Relu::~Relu() = default;


Conv2d::Conv2d(const std::vector<std::string> &parameters,
               const std::vector<std::vector<int>> &output_shape_list)
{
    /*
     * 分析参数为算子设置属性。可能的参数对包括：
     * input=%0
     * weight=../weight/conv1_weight.bin
     * bias=../weight/conv1_bias.bin
     * output_channel=16
     * input_channel=1
     * kernel_size=(3,3)
     * stride=(1,1)
     * padding=(1,1)
     * dilation=(1,1)
     */
    // 遍历参数对
    for(const std::string &s: parameters) {
        std::vector<std::string> para_pair = split(s, "=");
        if(para_pair[0] == "input") {
            std::string temp = replace(para_pair[1], "%", "");
            this->input_node = (int)strtol(temp.c_str(), nullptr, 10);
        }
        else if(para_pair[0] == "weight") {
            this->weight_path = (std::string)para_pair[1];  // 为什么加强制类型转换？我不知道，但不加Clion会标黄
        }
        else if(para_pair[0] == "bias") {
            this->bias_path = (std::string)para_pair[1];
        }
        else if(para_pair[0] == "output_channel") {
            this->output_channel = (int)strtol(para_pair[1].c_str(), nullptr, 10);
        }
        else if(para_pair[0] == "input_channel") {
            this->input_channel = (int)strtol(para_pair[1].c_str(), nullptr, 10);
        }
        else if(para_pair[0] == "kernel_size") {
            std::string temp = replace(para_pair[1], "(", "");
            temp = replace(temp, ")", "");
            std::vector<std::string> shape_str = split(temp, ",");
            for(const std::string &shape: shape_str) {
                kernel_size.push_back((int)strtol(shape.c_str(), nullptr, 10));
            }
        }
        else if(para_pair[0] == "stride") {
            std::string temp = replace(para_pair[1], "(", "");
            temp = replace(temp, ")", "");
            std::vector<std::string> shape_str = split(temp, ",");
            for(const std::string & shape: shape_str) {
                stride.push_back((int)strtol(shape.c_str(), nullptr, 10));
            }
        }
        else if(para_pair[0] == "padding") {
            std::string temp = replace(para_pair[1], "(", "");
            temp = replace(temp, ")", "");
            std::vector<std::string> shape_str = split(temp, ",");
            for(const std::string &shape: shape_str) {
                padding.push_back((int)strtol(shape.c_str(), nullptr, 10));
            }
        }
        else if(para_pair[0] == "dilation") {
            std::string temp = replace(para_pair[1], "(", "");
            temp = replace(temp, ")", "");
            std::vector<std::string> shape_str = split(temp, ",");
            for(const std::string &shape: shape_str) {
                dilation.push_back((int)strtol(shape.c_str(), nullptr, 10));
            }
        }
    }
    // 读取 weight
    weight = Tensor<float32>(std::vector<int>{output_channel, input_channel, kernel_size[0], kernel_size[1]});
    FILE * weight_file = fopen(weight_path.c_str(), "rb");
    fread(weight.data, sizeof(float), output_channel*input_channel*kernel_size[0]*kernel_size[1], weight_file);
    fclose(weight_file);
    // 读取 bias
    bias = Tensor<float32>(std::vector<int>{output_channel});
    FILE * bias_file = fopen(bias_path.c_str(), "rb");
    fread(bias.data, sizeof(float), output_channel, bias_file);
    fclose(bias_file);
    // 设置output_shape
    /*
     * 按如下方式计算output_shape，但不确定对不对
     * 0. NCHW中, N不变, C=output_channel，HW如下计算
     * 1. 根据padding，对上下左右进行扩展
     *      nh=h+ph*2, nw=w+ph*2
     * 2. 根据kernel_size和dilation计算卷积过程中，kernel覆盖的大小
     *      nkh=dh*(kh-1)+1, nkw=dw*(kw-1)+1
     * 3. 根据stride计算卷积过程中，卷积核截取图片的次数
     *      nh=(nh-nkh)/sh+1, nw=(nw-nkw)/sw+1
     */
    std::vector<int> input_shape = output_shape_list[input_node];
    output_shape.push_back(input_shape[0]);     // N
    output_shape.push_back(output_channel);     // C
    int h = input_shape[2];
    int w = input_shape[3];
    int nh = h + padding[0] * 2;
    int nw = w + padding[1] * 2;
    int nkh = dilation[0] * (kernel_size[0]-1) + 1;
    int nkw = dilation[1] * (kernel_size[1]-1) + 1;
    nh = (nh-nkh)/stride[0]+1;
    nw = (nw-nkw)/stride[1]+1;
    output_shape.push_back(nh);                 // H
    output_shape.push_back(nw);                 // W
}

void Conv2d::forward(Tensor<float32> *input, Tensor<float32> *output) {
    /*
     * Conv2d算子的forward
     */
    *output = F::conv2d(input, &weight, &bias, stride, padding, dilation);
}

Conv2d::~Conv2d() = default;


Maxpool2d::Maxpool2d(const std::vector<std::string> &parameters,
                     const std::vector<std::vector<int>> &output_shape_list)
{
    /*
     * 分析参数为算子设置属性。可能的参数对包括：
     * input=%2
     * kernel_size=(2,2)
     * stride=None or stride=(2,2)
     * padding=(0,0)
     * dilation=(1,1)
     */
    // 遍历参数对
    for(const std::string &s: parameters) {
        std::vector<std::string> para_pair = split(s, "=");
        if(para_pair[0] == "input") {
            std::string temp = replace(para_pair[1], "%", "");
            this->input_node = (int)strtol(temp.c_str(), nullptr, 10);
        }
        else if(para_pair[0] == "kernel_size") {
            std::string temp = replace(para_pair[1], "(", "");
            temp = replace(temp, ")", "");
            std::vector<std::string> shape_str = split(temp, ",");
            for(const std::string &shape: shape_str) {
                kernel_size.push_back((int)strtol(shape.c_str(), nullptr, 10));
            }
        }
        else if(para_pair[0] == "stride") {
            if(para_pair[1] == "None") {
                stride = kernel_size;
            }
            else {
                std::string temp = replace(para_pair[1], "(", "");
                temp = replace(temp, ")", "");
                std::vector<std::string> shape_str = split(temp, ",");
                for(const std::string &shape: shape_str) {
                    stride.push_back((int)strtol(shape.c_str(), nullptr, 10));
                }
            }
        }
        else if(para_pair[0] == "padding") {
            std::string temp = replace(para_pair[1], "(", "");
            temp = replace(temp, ")", "");
            std::vector<std::string> shape_str = split(temp, ",");
            for(const std::string &shape: shape_str) {
                padding.push_back((int)strtol(shape.c_str(), nullptr, 10));
            }
        }
        else if(para_pair[0] == "dilation") {
            std::string temp = replace(para_pair[1], "(", "");
            temp = replace(temp, ")", "");
            std::vector<std::string> shape_str = split(temp, ",");
            for(const std::string &shape: shape_str) {
                dilation.push_back((int)strtol(shape.c_str(), nullptr, 10));
            }
        }
    }
    // 设置output_shape
    /*
     * 0. NCHW中, N不变, C=output_channel，HW如下
     * 1. 根据padding，对上下左右进行扩展
     *      nh=h+ph*2, nw=w+ph*2
     * 2. 根据kernel_size和dilation计算池化过程中，kernel覆盖的大小
     *      nkh=dh*(kh-1)+1, nkw=dw*(kw-1)+1
     * 3. 根据stride计算池化过程中，kernel截取图片的次数
     *      nh=(nh-nkh)/sh+1, nw=(nw-nkw)/sw+1
     */
    std::vector<int> input_shape = output_shape_list[input_node];
    this->output_shape.push_back(input_shape[0]);       // N
    this->output_shape.push_back(input_shape[1]);       // C
    int h = input_shape[2];
    int w = input_shape[3];
    int nh = h+padding[0]*2;
    int nw = w+padding[1]*2;
    int nkh = dilation[0]*(kernel_size[0]-1)+1;
    int nkw = dilation[1]*(kernel_size[1]-1)+1;
    nh = (nh-nkh)/stride[0]+1;
    nw = (nw-nkw)/stride[1]+1;
    this->output_shape.push_back(nh);                   // H
    this->output_shape.push_back(nw);                   // W
}

void Maxpool2d::forward(Tensor<float32> *input, Tensor<float32> *output) {
    /*
     * Maxpool2d算子的forward
     */
    *output = F::maxpool2d(input, kernel_size, stride, padding, dilation);
}

Maxpool2d::~Maxpool2d() = default;

Flatten::Flatten(const std::vector<std::string> &parameters,
                 const std::vector<std::vector<int>> &output_shape_list)
{
    /*
     * 分析参数为算子设置属性。可能的参数对包括：
     * input=%1
     */
    // 遍历参数对
    for(const std::string &s: parameters) {
        std::vector<std::string> para_pair = split(s, "=");
        if(para_pair[0] == "input") {
            std::string temp = replace(para_pair[1], "%", "");
            this->input_node = (int)strtol(temp.c_str(), nullptr, 10);
        }
    }
    // 设置output_shape
    /*
     * NCHW中, N不变. CHW相乘变为L
     */
    std::vector<int> input_shape = output_shape_list[input_node];
    this->output_shape.push_back(input_shape[0]);
    int len = 1;
    for(int i = 1; i<(int)input_shape.size(); i++) {
        len *= input_shape[i];
    }
    this->output_shape.push_back(len);
}

void Flatten::forward(Tensor<float32> *input, Tensor<float32> *output) {
    /*
     * Flatten算子的forward
     */
    *output = F::flatten(input);
}

Flatten::~Flatten() = default;


Dense::Dense(const std::vector<std::string> &parameters,
             const std::vector<std::vector<int>> &output_shape_list)
{
    /*
     * 分析参数为算子设置属性。可能的参数对包括：
     * input=%9
     * weight=../weight/fc1_weight.bin
     * bias=../weight/fc1_bias.bin
     * output_channel=10
     * input_channel=3136
     */
    // 遍历参数对
    for(const std::string &s: parameters) {
        std::vector<std::string> para_pair = split(s, "=");
        if(para_pair[0] == "input") {
            std::string temp = replace(para_pair[1], "%", "");
            this->input_node = (int)strtol(temp.c_str(), nullptr, 10);
        }
        else if(para_pair[0] == "weight") {
            this->weight_path = (std::string)para_pair[1];
        }
        else if(para_pair[0] == "bias") {
            this->bias_path = (std::string)para_pair[1];
        }
        else if(para_pair[0] == "output_channel") {
            this->output_channel = (int)strtol(para_pair[1].c_str(), nullptr, 10);
        }
        else if(para_pair[0] == "input_channel") {
            this->input_channel = (int)strtol(para_pair[1].c_str(), nullptr, 10);
        }
    }
    // 读取 weight
    weight = Tensor<float32>(std::vector<int>{output_channel, input_channel});
    FILE * weight_file = fopen(weight_path.c_str(), "rb");
    fread(weight.data, sizeof(float), output_channel*input_channel, weight_file);
    fclose(weight_file);
    weight = weight.transpose(std::vector<int>{1,0});
    // 读取 bias
    bias = Tensor<float32>(std::vector<int>{output_channel});
    FILE * bias_file = fopen(bias_path.c_str(), "rb");
    fread(bias.data, sizeof(float), output_channel, bias_file);
    fclose(bias_file);
    // 设置output_shape
    /*
     * 在NL中, N不变, L变为output_channel
     */
    std::vector<int> input_shape = output_shape_list[input_node];
    output_shape.push_back(input_shape[0]);
    output_shape.push_back(output_channel);
}

void Dense::forward(Tensor<float32> *input, Tensor<float32> *output) {
    /*
     * Dense算子的forward
     */
    *output = F::dense(input, &weight, &bias);
}

Dense::~Dense() = default;

Output::Output(const std::vector<std::string> &parameters,
               const std::vector<std::vector<int> > &output_shape_list)
{
    /*
     * 分析参数为算子设置属性。可能的参数对包括：
     * input=%1
     */
    // 遍历参数对
    for(const std::string &s: parameters) {
        std::vector<std::string> para_pair = split(s, "=");
        if(para_pair[0] == "input") {
            std::string temp = replace(para_pair[1], "%", "");
            this->input_node = (int)strtol(temp.c_str(), nullptr, 10);
        }
    }
    // 设置output_shape
    this->output_shape = output_shape_list[input_node];
}

void Output::forward(Tensor<float32> *input, Tensor<float32> *output) {
    /*
     * Output算子forward
     */
    *output = (*input).deep_copy();
}

Output::~Output() = default;


Add::Add(const std::vector<std::string> &parameters,
         const std::vector<std::vector<int>> &output_shape_list)
{
    /*
     * 分析参数为算子设置属性。可能的参数对包括：
     * input1=%1
     * input2=%2
     */
    // 遍历参数对
    for(const std::string &s: parameters) {
        std::vector<std::string> para_pair = split(s, "=");
        if(para_pair[0] == "input1") {
            std::string temp = replace(para_pair[1], "%", "");
            this->input_node1 = (int)strtol(temp.c_str(), nullptr, 10);
        }
        else if(para_pair[0] == "input2") {
            std::string temp = replace(para_pair[1], "%", "");
            this->input_node2 = (int)strtol(temp.c_str(), nullptr, 10);
        }
    }
    // 设置output_shape
    assert(output_shape_list[input_node1] == output_shape_list[input_node2]);
    this->output_shape = output_shape_list[input_node1];
}

void Add::forward(Tensor<float32> *input1, Tensor<float32> *input2, Tensor<float32> *output) {
    /*
     * Add算子forward
     */
    *output = F::add(input1, input2);
}

Add::~Add() = default;

Concat::Concat(const std::vector<std::string> &parameters,
               const std::vector<std::vector<int>> &output_shape_list)
{
    /*
     * 分析参数为算子设置属性。可能的参数对包括：
     * input1=%1
     * input2=%2
     * dim=0
     */
    // 遍历参数对
    for(const std::string &s: parameters) {
        std::vector<std::string> para_pair = split(s, "=");
        if(para_pair[0] == "input1") {
            std::string temp = replace(para_pair[1], "%", "");
            this->input_node1 = (int)strtol(temp.c_str(), nullptr, 10);
        }
        else if(para_pair[0] == "input2") {
            std::string temp = replace(para_pair[1], "%", "");
            this->input_node2 = (int)strtol(temp.c_str(), nullptr, 10);
        }
        else if(para_pair[0] == "dim") {
            this->dim = (int)strtol(para_pair[1].c_str(), nullptr, 10);
        }
    }
    // 设置output_shape
    /*
     * dim指定的axis相加，其他不变
     */
    std::vector<int> input_shape1 = output_shape_list[input_node1];
    std::vector<int> input_shape2 = output_shape_list[input_node2];
    for(int i = 0; i<(int)input_shape1.size(); i++) {
        if(i == dim) {
            output_shape.push_back(input_shape1[i] + input_shape2[i]);
        }
        else {
            assert(input_shape1[i] == input_shape2[i]);
            output_shape.push_back(input_shape1[i]);
        }
    }
}

void Concat::forward(Tensor<float32> *input1, Tensor<float32> *input2, Tensor<float32> *output) {
    /*
     * concat算子forward
     */
    *output = F::concat(input1, input2, dim);
}

Concat::~Concat() = default;

Batch_Norm2d::Batch_Norm2d(const std::vector<std::string> &parameters,
                                         const std::vector<std::vector<int>> &output_shape_list)
{
    /*
     * 分析参数为算子设置属性
     * input=%1
     * num_features=64
     * eps=0.00001
     * momentum=0.1
     */
    // 遍历参数对
    for(const std::string &s: parameters) {
        std::vector<std::string> para_pair = split(s, "=");
        if(para_pair[0] == "input") {
            std::string temp = replace(para_pair[0], "%", "");
            this->input_node = (int)strtol(temp.c_str(), nullptr, 10);
        }
        else if(para_pair[0] == "num_features") {
            this->num_features = (int)strtol(para_pair[1].c_str(), nullptr, 10);
        }
        else if(para_pair[0] == "eps") {
            this->eps = (float)strtof(para_pair[1].c_str(), nullptr);
        }
        else if(para_pair[0] == "momentum") {
            this->momentum = (float)strtof(para_pair[1].c_str(), nullptr);
        }
    }
    // 设置output_shape
    this->output_shape = output_shape_list[input_node];
}

void Batch_Norm2d::forward(Tensor<float32> *input, Tensor<float32> *output) {
    /*
     * TODO: bn2d forward
     */
}

Batch_Norm2d::~Batch_Norm2d() = default;

