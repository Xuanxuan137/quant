//
// Created by noname on 2021/10/29.
//

#include "op.h"

Input::Input(const std::vector <std::string>& parameters)
{
    /*
     * Analyze parameters to set attributes for the Input operator. Possible parameter pairs:
     * shape=(1,1,28,28)
     * dtype=float32
     */
    // Traverse each parameter pair in parameters
    for(const std::string &s: parameters) {
        // Split parameter pair, divided into parameter and parameter value
        std::vector<std::string> para_pair = split(s, "=");
        // Different processing for each parameter
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
            // In the latest design, the data type and the operator name are bound, this parameter is no longer needed
        }
    }
}

//Input::~Input() {
//
//}

Relu::Relu(const std::vector<std::string>& parameters,
           const std::vector<std::vector<int> > &output_shape_list)
{
    /*
     * Analyze parameters to set attributes for Relu operator
     * input=%1
     */
    // Iterate over parameter pairs
    for(const std::string &s: parameters) {
        std::vector<std::string> para_pair = split(s, "=");
        if(para_pair[0] == "input") {
            std::string temp = replace(para_pair[1], "%", "");
            this->input_node = (int)strtol(temp.c_str(), nullptr, 10);
        }
    }
    // set output_shape
    output_shape = output_shape_list[input_node];
}


Conv2d::Conv2d(const std::vector<std::string> &parameters,
               const std::vector<std::vector<int>> &output_shape_list)
{
    /*
     * Analyze parameters to set attributes for Conv2d operator
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
    // Iterate over parameter pairs
    for(const std::string &s: parameters) {
        std::vector<std::string> para_pair = split(s, "=");
        if(para_pair[0] == "input") {
            std::string temp = replace(para_pair[1], "%", "");
            this->input_node = (int)strtol(temp.c_str(), nullptr, 10);
        }
        else if(para_pair[0] == "weight") {
            this->weight_path = (std::string)para_pair[1];  // Why add force type conversion? I don't know, but clion will mark it to yellow if I do not add
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
    // read weight
    weight = Vdarray<float>(std::vector<int>{output_channel, input_channel, kernel_size[0], kernel_size[1]});
    FILE * weight_file = fopen(weight_path.c_str(), "rb");
    fread(weight.data, sizeof(float), output_channel*input_channel*kernel_size[0]*kernel_size[1], weight_file);
    fclose(weight_file);
    // read bias
    bias = Vdarray<float>(std::vector<int>{output_channel});
    FILE * bias_file = fopen(bias_path.c_str(), "rb");
    fread(bias.data, sizeof(float), output_channel, bias_file);
    fclose(bias_file);
    // set output_shape
    /*
     * What should be calculated here? Currently calculated according to the following method, not sure if it is correct
     * 0. In NCHW, N will not change, C=output_channel，HW is calculated as follows
     * 1. According to padding，expand the input size up and down and left and right to get the padding size.
     *      nh=h+ph*2, nw=w+ph*2
     * 2. According to the kernel_size and dilation to calculate the kernel covers area in convolution.
     *      nkh=dh*(kh-1)+1, nkw=dw*(kw-1)+1
     * 3. Calculate the intercept time in convolution according to the kerneal cover area and stride
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


Maxpool2d::Maxpool2d(const std::vector<std::string> &parameters,
                     const std::vector<std::vector<int>> &output_shape_list)
{
    /*
     * Analyze parameters to set attributes for Maxpool2d operator
     * input=%2
     * kernel_size=(2,2)
     * stride=None or stride=(2,2)
     * padding=(0,0)
     * dilation=(1,1)
     */
    // Iterate over parameter pairs
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
    // set output_shape
    /*
     * What should be calculated here? Currently calculated according to the following method, not sure if it is correct
     * 0. In NCHW, N will not change, C=output_channel，HW is calculated as follows
     * 1. According to padding，expand the input size up and down and left and right to get the padding size.
     *      nh=h+ph*2, nw=w+ph*2
     * 2. According to the kernel_size and dilation to calculate the kernel covers area in convolution.
     *      nkh=dh*(kh-1)+1, nkw=dw*(kw-1)+1
     * 3. Calculate the intercept time in convolution according to the kerneal cover area and stride
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

Flatten::Flatten(const std::vector<std::string> &parameters,
                 const std::vector<std::vector<int>> &output_shape_list)
{
    /*
     * Analyze parameters to set attributes for Flatten operator
     * input=%1
     */
    // Iterate over parameter pairs
    for(const std::string &s: parameters) {
        std::vector<std::string> para_pair = split(s, "=");
        if(para_pair[0] == "input") {
            std::string temp = replace(para_pair[1], "%", "");
            this->input_node = (int)strtol(temp.c_str(), nullptr, 10);
        }
    }
    // set output_shape
    /*
     * In NCHW, N does not change. CHW multiplied to become L
     */
    std::vector<int> input_shape = output_shape_list[input_node];
    this->output_shape.push_back(input_shape[0]);
    int len = 1;
    for(int i = 1; i<(int)input_shape.size(); i++) {
        len *= input_shape[i];
    }
    this->output_shape.push_back(len);
}


Dense::Dense(const std::vector<std::string> &parameters,
             const std::vector<std::vector<int>> &output_shape_list)
{
    /*
     * Analyze parameters to set attributes for the Dense operator
     * input=%9
     * weight=../weight/fc1_weight.bin
     * bias=../weight/fc1_bias.bin
     * output_channel=10
     * input_channel=3136
     */
    // Iterate over parameter pairs
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
    // read weight
    weight = Vdarray<float>(std::vector<int>{output_channel, input_channel});
    FILE * weight_file = fopen(weight_path.c_str(), "rb");
    fread(weight.data, sizeof(float), output_channel*input_channel, weight_file);
    fclose(weight_file);
    // read bias
    bias = Vdarray<float>(std::vector<int>{output_channel});
    FILE * bias_file = fopen(bias_path.c_str(), "rb");
    fread(bias.data, sizeof(float), output_channel, bias_file);
    fclose(bias_file);
    // set output_shape
    /*
     * In NL, N does not change, L becomes output_channel
     */
    std::vector<int> input_shape = output_shape_list[input_node];
    output_shape.push_back(input_shape[0]);
    output_shape.push_back(output_channel);
}

Output::Output(const std::vector<std::string> &parameters,
               const std::vector<std::vector<int> > &output_shape_list)
{
    /*
     * Analyze parameters to set attributes for the Output operator
     * input=%1
     */
    // Iterate over parameter pairs
    for(const std::string &s: parameters) {
        std::vector<std::string> para_pair = split(s, "=");
        if(para_pair[0] == "input") {
            std::string temp = replace(para_pair[1], "%", "");
            this->input_node = (int)strtol(temp.c_str(), nullptr, 10);
        }
    }
    // set output_shape
    this->output_shape = output_shape_list[input_node];
}


Add::Add(const std::vector<std::string> &parameters,
         const std::vector<std::vector<int>> &output_shape_list)
{
    /*
     * Analyse parameters to set attributes for the Add operator
     * input1=%1
     * input2=%2
     */
    // Iterate over parameter pairs
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
    // set output_shape
    assert(output_shape_list[input_node1] == output_shape_list[input_node2]);
    this->output_shape = output_shape_list[input_node1];
}

Concat::Concat(const std::vector<std::string> &parameters,
               const std::vector<std::vector<int>> &output_shape_list)
{
    /*
     * Analyse parameters to set attributes for the Concat operator
     * input1=%1
     * input2=%2
     * dim=0
     */
    // Iterate over parameter pairs
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
    // Set output_shape
    /*
     * How to calculate output shape?
     * Add the axis specified by dim, others keep unchanged
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
