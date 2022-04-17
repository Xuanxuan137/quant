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
    if(input->size != output_shape) {
        fprintf(stderr, "File: op.cpp, line: %d. Cannot input data with shape (", __LINE__);
        for(const int &i: input->size) {
            fprintf(stderr, "%d, ", i);
        }
        fprintf(stderr, ") into Input Node with shape (");
        for(const int &i: output_shape) {
            fprintf(stderr, "%d, ", i);
        }
        fprintf(stderr, ")\n");
        exit(-1);
    }
    *output = *input;
}

void Input::print() {
    /*
     * 打印Input节点信息
     */
    std::string str;
    str += "input(shape=(";
    for(int i: output_shape) {
        char temp[10];
        sprintf(temp, "%d,", i);
        str += temp;
    }
    str.pop_back();
    str += "),dtype=\"float32\");";
    std::cout << str << std::endl;
}

void Input::save(const std::string &path, int number) {
    /*
     * 保存Input算子
     */
    char temp[20];
    sprintf(temp, "%%%d=", number);
    std::string str(temp);
    str += "input(shape=(";
    for(int i: output_shape) {
        sprintf(temp, "%d,", i);
        str += temp;
    }
    str.pop_back();
    str += "),dtype=\"float32\");\n";

    FILE * file = fopen((path+GRAPH_FILE_NAME).c_str(), "a");
    fprintf(file, "%s", str.c_str());
    fclose(file);
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

void Relu::print() {
    /*
     * 打印relu算子信息
     */
    std::string str;
    char temp[500];
    sprintf(temp, "nn.relu(input=%%%d, output_shape=(", input_node);
    str += temp;
    for(int i: output_shape) {
        sprintf(temp, "%d,", i);
        str += temp;
    }
    str.pop_back();
    str += "));\n";
    std::cout << str;
}

void Relu::save(const std::string &path, int number) {
    /*
     * 存储Relu算子信息
     */
    std::string str;
    char temp[500];
    sprintf(temp, "%%%d=nn.relu(input=%%%d, output_shape=(", number, input_node);
    str += temp;
    for(int i: output_shape) {
        sprintf(temp, "%d,", i);
        str += temp;
    }
    str.pop_back();
    str += "));\n";

    FILE * file = fopen((path+GRAPH_FILE_NAME).c_str(), "a");
    fprintf(file, "%s", str.c_str());
    fclose(file);
}

Relu::~Relu() = default;


Conv2d::Conv2d(const std::vector<std::string> &parameters,
               const std::vector<std::vector<int> > &output_shape_list)
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

void Maxpool2d::print() {
    /*
     * 打印maxpool2d算子信息
     */
    char temp[500];
    sprintf(temp, "nn.maxpool2d(input=%%%d, kernel_size=(%d,%d), stride=(%d,%d), "
                  "padding=(%d,%d), dilation=(%d,%d), output_shape=(%d,%d,%d,%d));\n",
            input_node, kernel_size[0], kernel_size[1], stride[0], stride[1], padding[0],
            padding[1], dilation[0], dilation[1], output_shape[0], output_shape[1],
            output_shape[2], output_shape[3]);
    printf("%s", temp);
}

void Maxpool2d::save(const std::string &path, int number) {
    /*
     * 存储maxpool2d算子信息
     */
    char temp[500];
    sprintf(temp, "%%%d=nn.maxpool2d(input=%%%d, kernel_size=(%d,%d), stride=(%d,%d), "
                  "padding=(%d,%d), dilation=(%d,%d), output_shape=(%d,%d,%d,%d));\n",
            number, input_node, kernel_size[0], kernel_size[1], stride[0], stride[1], padding[0],
            padding[1], dilation[0], dilation[1], output_shape[0], output_shape[1],
            output_shape[2], output_shape[3]);

    FILE * file = fopen((path+GRAPH_FILE_NAME).c_str(), "a");
    fprintf(file, "%s", temp);
    fclose(file);
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

void Flatten::print() {
    /*
     * 打印flatten算子信息
     */
    char temp[500];
    sprintf(temp, "nn.flatten(input=%%%d, output_shape=(%d,%d));\n",
            input_node, output_shape[0], output_shape[1]);
    printf("%s", temp);
}

void Flatten::save(const std::string &path, int number) {
    /*
     * 存储Flatten算子
     */
    char temp[500];
    sprintf(temp, "%%%d=nn.flatten(input=%%%d, output_shape=(%d,%d));\n",
            number, input_node, output_shape[0], output_shape[1]);

    FILE * file = fopen((path+GRAPH_FILE_NAME).c_str(), "a");
    fprintf(file, "%s", temp);
    fclose(file);
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

void Dense::print() {
    /*
     * 打印dense算子信息
     */
    char temp[500];
    sprintf(temp, "nn.dense(input=%%%d, output_channel=%d, input_channel=%d, "
                  "output_shape=(%d,%d));\n",
                  input_node, output_channel, input_channel, output_shape[0], output_shape[1]);
    printf("%s", temp);
}

void Dense::save(const std::string &path, int number) {
    /*
     * 存储Dense算子信息
     */
    char temp[500];
    char save_weight_path[200];
    char save_bias_path[200];
    sprintf(save_weight_path, "%sdense_%d_weight.bin", path.c_str(), number);
    sprintf(save_bias_path, "%sdense_%d_bias.bin", path.c_str(), number);
    sprintf(temp, "%%%d=nn.dense(input=%%%d, weight=%s, bias=%s, output_channel=%d, input_channel=%d, "
                  "output_shape=(%d,%d));\n",
            number, input_node, save_weight_path, save_bias_path, output_channel, input_channel,
            output_shape[0], output_shape[1]);

    FILE * file = fopen((path+GRAPH_FILE_NAME).c_str(), "a");
    fprintf(file, "%s", temp);
    fclose(file);

    // 存储weight bias
    FILE * wf = fopen(save_weight_path, "wb");
    fwrite(weight.data, sizeof(float32), weight.len(), wf);
    fclose(wf);
    FILE * bf = fopen(save_bias_path, "wb");
    fwrite(bias.data, sizeof(float32), bias.len(), bf);
    fclose(bf);
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

void Output::print() {
    /*
     * 打印output算子信息
     */
    std::string str;
    char temp[500];
    sprintf(temp, "output(input=%%%d, output_shape=(", input_node);
    str += temp;
    for(int i: output_shape) {
        sprintf(temp, "%d,", i);
        str += temp;
    }
    str.pop_back();
    str += "));\n";
    std::cout << str;
}

void Output::save(const std::string &path, int number) {
    /*
     * 存储Output算子信息
     */
    std::string str;
    char temp[500];
    sprintf(temp, "%%%d=output(input=%%%d, output_shape=(", number, input_node);
    str += temp;
    for(int i: output_shape) {
        sprintf(temp, "%d,", i);
        str += temp;
    }
    str.pop_back();
    str += "));\n";

    FILE * file = fopen((path+GRAPH_FILE_NAME).c_str(), "a");
    fprintf(file, "%s", str.c_str());
    fclose(file);
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

void Add::print() {
    /*
     * 打印add算子信息
     */
    std::string str;
    char temp[500];
    sprintf(temp, "add(input1=%%%d, input2=%%%d, output_shape=(", input_node1, input_node2);
    str += temp;
    for(int i: output_shape) {
        sprintf(temp, "%d,", i);
        str += temp;
    }
    str.pop_back();
    str += "));\n";
    std::cout << str;
}

void Add::save(const std::string &path, int number) {
    /*
     * 存储Add算子信息
     */
    char temp[500];
    sprintf(temp, "%%%d=add(input1=%%%d, input2=%%%d, output_shape=(", number, input_node1, input_node2);
    std::string str(temp);
    for(int i: output_shape) {
        sprintf(temp, "%d,", i);
        str += temp;
    }
    str.pop_back();
    str += "));\n";

    FILE * file = fopen((path+GRAPH_FILE_NAME).c_str(), "a");
    fprintf(file, "%s", str.c_str());
    fclose(file);
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

void Concat::print() {
    /*
     * 打印concat算子信息
     */
    std::string str;
    char temp[500];
    sprintf(temp, "concat(input1=%%%d, input2=%%%d, dim=%d, output_shape=(", input_node1, input_node2, dim);
    str += temp;
    for(int i: output_shape) {
        sprintf(temp, "%d,", i);
        str += temp;
    }
    str.pop_back();
    str += "));\n";
    std::cout << str;
}

void Concat::save(const std::string &path, int number) {
    /*
     * 存储Concat算子信息
     */
    char temp[500];
    sprintf(temp, "%%%d=concat(input1=%%%d, input2=%%%d, dim=%d, output_shape=(",
            number, input_node1, input_node2, dim);
    std::string str(temp);
    for(int i: output_shape) {
        sprintf(temp, "%d,", i);
        str += temp;
    }
    str.pop_back();
    str += "));\n";

    FILE * file = fopen((path+GRAPH_FILE_NAME).c_str(), "a");
    fprintf(file, "%s", str.c_str());
    fclose(file);
}

Concat::~Concat() = default;

Batch_Norm2d::Batch_Norm2d(const std::vector<std::string> &parameters,
                                         const std::vector<std::vector<int>> &output_shape_list)
{
    /*
     * 分析参数为算子设置属性
     * input=%1
     * weight=../weight/bn1_weight.bin
     * bias=../weight/bn1_bias.bin
     * num_features=64
     * eps=0.00001
     * momentum=0.1
     */
    // 遍历参数对
    for(const std::string &s: parameters) {
        std::vector<std::string> para_pair = split(s, "=");
        if(para_pair[0] == "input") {
            std::string temp = replace(para_pair[1], "%", "");
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
        else if(para_pair[0] == "weight") {
            this->weight_path = (std::string)para_pair[1];
        }
        else if(para_pair[0] == "bias") {
            this->bias_path = (std::string)para_pair[1];
        }
    }
    // 读取weight
    weight = Tensor<float32>{std::vector<int>{num_features}};
    FILE * weight_file = fopen(weight_path.c_str(), "rb");
    fread(weight.data, sizeof(float), num_features, weight_file);
    fclose(weight_file);
    // 读取bias
    bias = Tensor<float32>{std::vector<int>{num_features}};
    FILE * bias_file = fopen(bias_path.c_str(), "rb");
    fread(bias.data, sizeof(float), num_features, bias_file);
    fclose(bias_file);
    // 设置output_shape
    this->output_shape = output_shape_list[input_node];
}

void Batch_Norm2d::forward(Tensor<float32> *input, Tensor<float32> *output) {
    /*
     * TODO: bn2d forward
     */
    // 根据input计算running mean和running var
    Tensor<float32> running_mean = input->mean(std::vector<int>{0,2,3});
    Tensor<float32> running_var = input->var(std::vector<int>{0,2,3});
    // 调用Functional
    *output = F::batch_norm2d(input, &running_mean, &running_var, &weight, &bias, eps, momentum);
}

void Batch_Norm2d::print() {
    /*
     * 打印batch_norm2d算子信息
     */
    char temp[500];
    sprintf(temp, "nn.batch_nor2d(input=%%%d, num_features=%d, eps=%f, "
                  "momentum=%f, output_shape=(%d,%d,%d,%d));\n",
                  input_node, num_features, eps, momentum, output_shape[0], output_shape[1],
                  output_shape[2], output_shape[3]);
    printf("%s", temp);
}

void Batch_Norm2d::save(const std::string &path, int number) {
    /*
     * 存储Batch_Norm2d算子信息
     */
    char temp[500];
    sprintf(temp, "%%%d=nn.batch_nor2d(input=%%%d, num_features=%d, eps=%f, "
                  "momentum=%f, output_shape=(%d,%d,%d,%d));\n",
            number, input_node, num_features, eps, momentum, output_shape[0], output_shape[1],
            output_shape[2], output_shape[3]);

    FILE * file = fopen((path+GRAPH_FILE_NAME).c_str(), "a");
    fprintf(file, "%s", temp);
    fclose(file);
}

Batch_Norm2d::~Batch_Norm2d() = default;

void Conv2d::print() {
    /*
     * 打印conv2d信息
     */
    char temp[500];
    sprintf(temp, "nn.conv2d(input=%%%d, output_channel=%d, input_channel=%d,"
                  "kernel_size=(%d,%d), stride=(%d,%d), padding=(%d,%d), dilation=(%d,%d), "
                  "output_shape=(%d,%d,%d,%d));\n",
                  input_node, output_channel, input_channel, kernel_size[0], kernel_size[1],
                  stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
                  output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
    printf("%s", temp);
}

void Conv2d::save(const std::string &path, int number) {
    /*
     * 存储conv2d信息
     */
    char temp[1000];
    char save_weight_path[200];
    char save_bias_path[200];
    sprintf(save_weight_path, "%sconv2d_%d_weight.bin", path.c_str(), number);
    sprintf(save_bias_path, "%sconv2d_%d_bias.bin", path.c_str(), number);
    sprintf(temp, "%%%d=nn.conv2d(input=%%%d, weight=%s, bias=%s, output_channel=%d, input_channel=%d,"
                  "kernel_size=(%d,%d), stride=(%d,%d), padding=(%d,%d), dilation=(%d,%d), "
                  "output_shape=(%d,%d,%d,%d));\n",
            number, input_node, save_weight_path, save_bias_path, output_channel, input_channel,
            kernel_size[0], kernel_size[1], stride[0], stride[1], padding[0], padding[1],
            dilation[0], dilation[1], output_shape[0], output_shape[1], output_shape[2], output_shape[3]);

    FILE * file = fopen((path+GRAPH_FILE_NAME).c_str(), "a");
    fprintf(file, "%s", temp);
    fclose(file);

    // 存储weight bias
    FILE * wf = fopen(save_weight_path, "wb");
    fwrite(weight.data, sizeof(float32), weight.len(), wf);
    fclose(wf);
    FILE * bf = fopen(save_bias_path, "wb");
    fwrite(bias.data, sizeof(float32), bias.len(), bf);
    fclose(bf);
}

QConv2d::QConv2d(Conv2d *op) {
    /*
     * QConv2d构造函数。由于量化算子是由普通算子量化得到的，而非直接从计算图中读取到的，
     * 因此使用conv2d算子进行构建。其他量化算子与此类似
     */
    input_node = op->input_node;
    weight = Tensor<uint8>{op->weight.size};
    bias = Tensor<int32>{op->bias.size};
    output_channel = op->output_channel;
    input_channel = op->input_channel;
    kernel_size = op->kernel_size;
    stride = op->stride;
    padding = op->padding;
    dilation = op->dilation;
    output_shape = op->output_shape;
    zero_x = 0;
    zero_w = 0;
    zero_b = 0;
    zero_y = 0;
    coe = 0;
    rshift = 0;
    qmin = 0;
    qmax = 0;
}

void QConv2d::print() {
    /*
     * 打印qconv2d信息
     */
    char temp[500];
    sprintf(temp, "nn.qconv2d(input=%%%d, output_channel=%d, input_channel=%d,"
                  "kernel_size=(%d,%d), stride=(%d,%d), padding=(%d,%d), dilation=(%d,%d), "
                  "output_shape=(%d,%d,%d,%d));\n",
            input_node, output_channel, input_channel, kernel_size[0], kernel_size[1],
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
            output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
    printf("%s", temp);
}

void QConv2d::forward(Tensor<uint8> *input, Tensor<uint8> *output) {
    /*
     * QConv2d前向传播函数
     */
    *output = F::qconv2d(input, zero_x, zero_w, zero_b, zero_y, coe, rshift, qmin, qmax,
                         &weight, &bias, stride, padding, dilation);
}

void QConv2d::save(const std::string &path, int number) {
    /*
     * 存储QConv2d信息
     */
    char temp[2000];
    char save_weight_path[200];
    char save_bias_path[200];
    sprintf(save_weight_path, "%sqconv2d_%d_weight.bin", path.c_str(), number);
    sprintf(save_bias_path, "%sqconv2d_%d_bias.bin", path.c_str(), number);
    sprintf(temp, "%%%d=nn.qconv2d(input=%%%d, weight=%s, bias=%s, output_channel=%d, input_channel=%d,"
                  "kernel_size=(%d,%d), stride=(%d,%d), padding=(%d,%d), dilation=(%d,%d), "
                  "output_shape=(%d,%d,%d,%d), zero_x=%d, zero_w=%d, zero_b=%d, zero_y=%d, "
                  "coe=%f, rshift=%d, qmin=%d, qmax=%d);\n",
            number, input_node, save_weight_path, save_bias_path, output_channel, input_channel,
            kernel_size[0], kernel_size[1], stride[0], stride[1], padding[0], padding[1],
            dilation[0], dilation[1], output_shape[0], output_shape[1], output_shape[2], output_shape[3],
            zero_x, zero_w, zero_b, zero_y, coe.get_value(), rshift, qmin, qmax);

    FILE * file = fopen((path+GRAPH_FILE_NAME).c_str(), "a");
    fprintf(file, "%s", temp);
    fclose(file);

    bias.print();

    // 存储weight bias
    FILE * wf = fopen(save_weight_path, "wb");
    fwrite(weight.data, sizeof(uint8), weight.len(), wf);
    fclose(wf);
    FILE * bf = fopen(save_bias_path, "wb");
    fwrite(bias.data, sizeof(int32), bias.len(), bf);
    fclose(bf);
}

QConv2d::~QConv2d() = default;

QInput::QInput(Input *op) {
    /*
     * 创建QInput算子
     */
    output_shape = op->output_shape;
}

void QInput::print() {
    /*
     * 打印qInput节点信息
     */
    std::string str;
    str += "qinput(shape=(";
    for(int i: output_shape) {
        char temp[10];
        sprintf(temp, "%d,", i);
        str += temp;
    }
    str.pop_back();
    str += "),dtype=\"uint8\");";
    std::cout << str<< std::endl;
}

void QInput::forward(Tensor<uint8> *input, Tensor<uint8> *output) {
    /*
     * QInput前向传播函数
     */
    if(input->size != output_shape) {
        fprintf(stderr, "File: op.cpp, line: %d. Cannot input data with shape (", __LINE__);
        for(const int &i: input->size) {
            fprintf(stderr, "%d, ", i);
        }
        fprintf(stderr, ") into Input Node with shape (");
        for(const int &i: output_shape) {
            fprintf(stderr, "%d, ", i);
        }
        fprintf(stderr, ")\n");
        exit(-1);
    }
    *output = *input;
}

void QInput::save(const std::string &path, int number) {
    /*
     * 保存QInput算子
     */
    char temp[100];
    sprintf(temp, "%%%d=", number);
    std::string str(temp);
    str += "qinput(shape=(";
    for(int i: output_shape) {
        sprintf(temp, "%d,", i);
        str += temp;
    }
    str.pop_back();
    str += "),dtype=\"uint8\");\n";

    FILE * file = fopen((path+GRAPH_FILE_NAME).c_str(), "a");
    fprintf(file, "%s", str.c_str());
    fclose(file);
}

QInput::~QInput() = default;

QMaxpool2d::QMaxpool2d(Maxpool2d *op) {
    /*
     * 创建QMaxpool2d算子
     */
    input_node = op->input_node;
    kernel_size = op->kernel_size;
    stride = op->stride;
    padding = op->padding;
    dilation = op->dilation;
    output_shape = op->output_shape;
    zero = 0;
}

void QMaxpool2d::print() {
    /*
     * 打印maxpool2d算子信息
     */
    char temp[500];
    sprintf(temp, "nn.qmaxpool2d(input=%%%d, kernel_size=(%d,%d), stride=(%d,%d), "
                  "padding=(%d,%d), dilation=(%d,%d), output_shape=(%d,%d,%d,%d));\n",
            input_node, kernel_size[0], kernel_size[1], stride[0], stride[1], padding[0],
            padding[1], dilation[0], dilation[1], output_shape[0], output_shape[1],
            output_shape[2], output_shape[3]);
    printf("%s", temp);
}

void QMaxpool2d::forward(Tensor<uint8> *input, Tensor<uint8> *output) {
    /*
     * QMaxpool2d前向传播韩函数
     */
    *output = F::qmaxpool2d(input, zero, kernel_size, stride, padding, dilation);
}

void QMaxpool2d::save(const std::string &path, int number) {
    /*
     * 存储QMaxpool2d算子信息
     */
    char temp[500];
    sprintf(temp, "%%%d=nn.qmaxpool2d(input=%%%d, kernel_size=(%d,%d), stride=(%d,%d), "
                  "padding=(%d,%d), dilation=(%d,%d), output_shape=(%d,%d,%d,%d), zero=%d);\n",
            number, input_node, kernel_size[0], kernel_size[1], stride[0], stride[1], padding[0],
            padding[1], dilation[0], dilation[1], output_shape[0], output_shape[1],
            output_shape[2], output_shape[3], zero);

    FILE * file = fopen((path+GRAPH_FILE_NAME).c_str(), "a");
    fprintf(file, "%s", temp);
    fclose(file);
}

QMaxpool2d::~QMaxpool2d() = default;

QRelu::QRelu(Relu *op) {
    /*
     * 创建QRelu算子
     */
    input_node = op->input_node;
    output_shape = op->output_shape;
    zero = 0;
    qmax = 0;
}

void QRelu::print() {
    /*
     * 打印relu算子信息
     */
    std::string str;
    char temp[500];
    sprintf(temp, "nn.qrelu(input=%%%d, output_shape=(", input_node);
    str += temp;
    for(int i: output_shape) {
        sprintf(temp, "%d,", i);
        str += temp;
    }
    str.pop_back();
    str += "));\n";
    std::cout << str;
}

void QRelu::forward(Tensor<uint8> *input, Tensor<uint8> *output) {
    /*
     * QRelu前向传播函数
     */
    *output = F::qrelu(input, zero, qmax);
}

void QRelu::save(const std::string &path, int number) {
    /*
     * 存储QRelu算子信息
     */
    std::string str;
    char temp[500];
    sprintf(temp, "%%%d=nn.qrelu(input=%%%d, output_shape=(", number, input_node);
    str += temp;
    for(int i: output_shape) {
        sprintf(temp, "%d,", i);
        str += temp;
    }
    str.pop_back();
    sprintf(temp, "%s), zero=%d, qmax=%d)\n", str.c_str(), zero, qmax);

    FILE * file = fopen((path+GRAPH_FILE_NAME).c_str(), "a");
    fprintf(file, "%s", temp);
    fclose(file);
}

QRelu::~QRelu() = default;

QFlatten::QFlatten(Flatten *op) {
    /*
     * 创建QFlatten算子
     */
    input_node = op->input_node;
    output_shape = op->output_shape;
}

void QFlatten::print() {
    /*
     * 打印flatten算子信息
     */
    char temp[500];
    sprintf(temp, "nn.qflatten(input=%%%d, output_shape=(%d,%d));\n",
            input_node, output_shape[0], output_shape[1]);
    printf("%s", temp);
}

void QFlatten::forward(Tensor<uint8> *input, Tensor<uint8> *output) {
    /*
     * QFlatten前向传播函数
     */
    *output = F::qflatten(input);
}

void QFlatten::save(const std::string &path, int number) {
    /*
     * 存储QFlatten算子信息
     */
    char temp[500];
    sprintf(temp, "%%%d=nn.qflatten(input=%%%d, output_shape=(%d,%d));\n",
            number, input_node, output_shape[0], output_shape[1]);

    FILE * file = fopen((path+GRAPH_FILE_NAME).c_str(), "a");
    fprintf(file, "%s", temp);
    fclose(file);
}

QFlatten::~QFlatten() = default;

QDense::QDense(Dense *op) {
    /*
     * 创建QDense算子
     */
    input_node = op->input_node;
    weight = Tensor<uint8>{op->weight.size};
    bias = Tensor<int32>{op->bias.size};
    output_channel = op->output_channel;
    input_channel = op->input_channel;
    output_shape = op->output_shape;
    zero_x = 0;
    zero_w = 0;
    zero_b = 0;
    zero_y = 0;
    coe = 0;
    rshift = 0;
    qmin = 0;
    qmax = 0;
}

void QDense::print() {
    /*
     * 打印qdense算子信息
     */
    char temp[500];
    sprintf(temp, "nn.qdense(input=%%%d, output_channel=%d, input_channel=%d, "
                  "output_shape=(%d,%d));\n",
            input_node, output_channel, input_channel, output_shape[0], output_shape[1]);
    printf("%s", temp);
}

void QDense::forward(Tensor<uint8> *input, Tensor<uint8> *output) {
    /*
     * QDense前向传播函数
     */
    *output = F::qdense(input, zero_x, zero_w, zero_b, zero_y, coe, rshift, qmin, qmax,
                        &weight, &bias);
}

void QDense::save(const std::string &path, int number) {
    /*
     * 存储QDense算子信息
     */
    char temp[1000];
    char save_weight_path[200];
    char save_bias_path[200];
    sprintf(save_weight_path, "%sqdense_%d_weight.bin", path.c_str(), number);
    sprintf(save_bias_path, "%sqdense_%d_bias.bin", path.c_str(), number);
    sprintf(temp, "%%%d=nn.qdense(input=%%%d, weight=%s, bias=%s, output_channel=%d, input_channel=%d, "
                  "output_shape=(%d,%d), zero_x=%d, zero_w=%d, zero_b=%d, zero_y=%d, "
                  "coe=%f, rshift=%d, qmin=%d, qmax=%d);\n",
            number, input_node, save_weight_path, save_bias_path, output_channel, input_channel,
            output_shape[0], output_shape[1], zero_x, zero_w, zero_b, zero_y,
            coe.get_value(), rshift, qmin, qmax);

    FILE * file = fopen((path+GRAPH_FILE_NAME).c_str(), "a");
    fprintf(file, "%s", temp);
    fclose(file);

    // 存储weight bias
    FILE * wf = fopen(save_weight_path, "wb");
    fwrite(weight.data, sizeof(uint8), weight.len(), wf);
    fclose(wf);
    FILE * bf = fopen(save_bias_path, "wb");
    fwrite(bias.data, sizeof(int32), bias.len(), bf);
    fclose(bf);
}

QDense::~QDense() = default;

QOutput::QOutput(Output *op) {
    /*
     * 创建QOutput算子
     */
    input_node = op->input_node;
    output_shape = op->output_shape;
}

void QOutput::print() {
    /*
     * 打印output算子信息
     */
    std::string str;
    char temp[500];
    sprintf(temp, "qoutput(input=%%%d, output_shape=(", input_node);
    str += temp;
    for(int i: output_shape) {
        sprintf(temp, "%d,", i);
        str += temp;
    }
    str.pop_back();
    str += "));\n";
    std::cout << str;
}

void QOutput::forward(Tensor<uint8> *input, Tensor<uint8> *output) {
    /*
     * QOutput前向传播函数
     */
    *output = (*input).deep_copy();
}

void QOutput::save(const std::string &path, int number) {
    /*
     * 存储QOutput算子参数
     */
    std::string str;
    char temp[500];
    sprintf(temp, "%%%d=qoutput(input=%%%d, output_shape=(", number, input_node);
    str += temp;
    for(int i: output_shape) {
        sprintf(temp, "%d,", i);
        str += temp;
    }
    str.pop_back();
    str += "));\n";

    FILE * file = fopen((path+GRAPH_FILE_NAME).c_str(), "a");
    fprintf(file, "%s", str.c_str());
    fclose(file);
}

QOutput::~QOutput() = default;

QAdd::QAdd(Add *op) {
    /*
     * 创建QAdd算子
     */
    input_node1 = op->input_node1;
    input_node2 = op->input_node2;
    output_shape = op->output_shape;
    zero_x1 = 0;
    zero_x2 = 0;
    zero_y = 0;
    coe1 = 0;
    coe2 = 0;
    rshift1 = 0;
    rshift2 = 0;
}

void QAdd::print() {
    /*
     * 打印qadd算子信息
     */
    std::string str;
    char temp[500];
    sprintf(temp, "qadd(input1=%%%d, input2=%%%d, output_shape=(", input_node1, input_node2);
    str += temp;
    for(int i: output_shape) {
        sprintf(temp, "%d,", i);
        str += temp;
    }
    str.pop_back();
    str += "));\n";
    std::cout << str;
}

void QAdd::forward(Tensor<uint8> *input1, Tensor<uint8> *input2, Tensor<uint8> *output) {
    /*
     * QAdd前向传播函数
     */
    *output = F::qadd(input1, input2, zero_x1, zero_x2, zero_y, coe1, coe2, rshift1, rshift1, qmin, qmax);
}

void QAdd::save(const std::string &path, int number) {
    /*
     * 存储QAdd算子信息
     */
    char temp[500];
    sprintf(temp, "%%%d=qadd(input1=%%%d, input2=%%%d, output_shape=(", number, input_node1, input_node2);
    std::string str(temp);
    for(int i: output_shape) {
        sprintf(temp, "%d,", i);
        str += temp;
    }
    str.pop_back();
    sprintf(temp, "), zero_x1=%d, zero_x2=%d, zero_y=%d, coe1=%f, coe2=%f, rshift1=%d, rshift2=%d, "
                  "qmin=%d, qmax=%d);\n",
                  zero_x1, zero_x2, zero_y, coe1.get_value(), coe2.get_value(), rshift1, rshift2, qmin, qmax);
    str += temp;

    FILE * file = fopen((path+GRAPH_FILE_NAME).c_str(), "a");
    fprintf(file, "%s", str.c_str());
    fclose(file);
}


QAdd::~QAdd() = default;

QConcat::QConcat(Concat *op) {
    /*
     * 创建QConcat算子
     */
    input_node1 = op->input_node1;
    input_node2 = op->input_node2;
    dim = op->dim;
    output_shape = op->output_shape;
    zero_x1 = 0;
    zero_x2 = 0;
    zero_y = 0;
    coe1 = 0;
    coe2 = 0;
    rshift1 = 0;
    rshift2 = 0;
    qmin = 0;
    qmax = 0;
}

void QConcat::print() {
    /*
     * 打印concat算子信息
     */
    std::string str;
    char temp[500];
    sprintf(temp, "qconcat(input1=%%%d, input2=%%%d, dim=%d, output_shape=(", input_node1, input_node2, dim);
    str += temp;
    for(int i: output_shape) {
        sprintf(temp, "%d,", i);
        str += temp;
    }
    str.pop_back();
    str += "));\n";
    std::cout << str;
}

void QConcat::forward(Tensor<uint8> *input1, Tensor<uint8> *input2, Tensor<uint8> *output) {
    /*
     * QConcat前向传播函数
     */
    *output = F::qconcat(input1, input2, zero_x1, zero_x2, zero_y, coe1, coe2, rshift1, rshift2, qmin, qmax, dim);
}

void QConcat::save(const std::string &path, int number) {
    /*
     * 存储QConcat算子信息
     */
    std::string str;
    char temp[500];
    sprintf(temp, "%%%d=qconcat(input1=%%%d, input2=%%%d, dim=%d, output_shape=(",
            number, input_node1, input_node2, dim);
    str += temp;
    for(int i: output_shape) {
        sprintf(temp, "%d,", i);
        str += temp;
    }
    str.pop_back();
    sprintf(temp, "), zero_x1=%d, zero_x2=%d, zero_y=%d, coe1=%f, coe2=%f, rshif1=%d, rshif2=%d, "
                  "qmin=%d, qmax=%d);\n",
                  zero_x1, zero_x2, zero_y, coe1.get_value(), coe2.get_value(), rshift1, rshift2, qmin, qmax);
    str += temp;

    FILE * file = fopen((path+GRAPH_FILE_NAME).c_str(), "a");
    fprintf(file, "%s", str.c_str());
    fclose(file);
}

QConcat::~QConcat() = default;
