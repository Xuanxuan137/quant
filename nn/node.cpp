//
// Created by noname on 2021/10/28.
//

#include "node.h"

Node::Node(const std::string& read_graph_line,
           const std::vector<std::vector<int> > &output_shape_list) {
    /*
     * Constructor:
     * 根据输入的计算图的一行
     * 1. 提取编号，为number属性赋值
     * 2. 提取算子名称，为name属性赋值
     * 3. 根据算子名称，设置dtype属性
     * 4. 根据算子类型创建算子对象，并传入计算图的一行，使其初始化
     * 5. 根据算子的output_shape属性，设置node的output_shape属性
     */
    std::string graph_line = replace(read_graph_line, " ", "");

    // 提取编号
    this->number = get_number(graph_line);
    // 提取算子名称
    this->name = get_name(graph_line);
    // 提取算子参数. 分割参数, 每个存为一个string, 并存在一个vector里
    std::vector<std::string> parameters = get_parameters(graph_line);
    // 根据算子名称创建对象
    printf("%d\n", this->name);
    if(this->name == OPN_NN_CONV2D) {
        this->dtype = "float32";
        op = new Conv2d(parameters, output_shape_list);
        this->output_shape = ((Conv2d*)op)->output_shape;
    }
    else if(this->name == OPN_NN_RELU) {
        this->dtype = "float32";
        op = new Relu(parameters, output_shape_list);
        this->output_shape = ((Relu*)op)->output_shape;
    }
    else if(this->name == OPN_NN_MAXPOOL2D) {
        this->dtype = "float32";
        op = new Maxpool2d(parameters, output_shape_list);
        this->output_shape = ((Maxpool2d*)op)->output_shape;
    }
    else if(this->name == OPN_INPUT) {
        this->dtype = "float32";
        op = new Input(parameters);
        this->output_shape = ((Input*)op)->output_shape;
    }
    else if(this->name == OPN_NN_FLATTEN) {
        this->dtype = "float32";
        op = new Flatten(parameters, output_shape_list);
        this->output_shape = ((Flatten*)op)->output_shape;
    }
    else if(this->name == OPN_NN_DENSE) {
        this->dtype = "float32";
        op = new Dense(parameters, output_shape_list);
        this->output_shape = ((Dense*)op)->output_shape;
    }
    else if(this->name == OPN_OUTPUT) {
        this->dtype = "float32";
        op = new Output(parameters, output_shape_list);
        this->output_shape = ((Output*)op)->output_shape;
    }
    else if(this->name == OPN_ADD) {
        this->dtype = "float32";
        op = new Add(parameters, output_shape_list);
        this->output_shape = ((Add*)op)->output_shape;
    }
    else if(this->name == OPN_CONCAT) {
        this->dtype = "float32";
        op = new Concat(parameters, output_shape_list);
        this->output_shape = ((Concat*)op)->output_shape;
    }
    else if(this->name == OPN_NN_BATCH_NORM2D) {
        this->dtype = "float32";
        op = new Batch_Norm2d(parameters, output_shape_list);
        this->output_shape = ((Batch_Norm2d*)op)->output_shape;
    }
}

Node::~Node() {
    /*
     * destructor: 释放op
     */
    // 算子
    if(this->name == OPN_NN_CONV2D) {
        delete((Conv2d*)op);
    }
    else if(this->name == OPN_NN_RELU) {
        delete((Relu*)op);
    }
    else if(this->name == OPN_INPUT) {
        delete((Input*)op);
    }
    else if(this->name == OPN_NN_MAXPOOL2D) {
        delete((Maxpool2d*)op);
    }
    else if(this->name == OPN_NN_FLATTEN) {
        delete((Flatten*)op);
    }
    else if(this->name == OPN_NN_DENSE) {
        delete((Dense*)op);
    }
    else if(this->name == OPN_OUTPUT) {
        delete((Output*)op);
    }
    else if(this->name == OPN_ADD) {
        delete((Add*)op);
    }
    else if(this->name == OPN_CONCAT) {
        delete((Concat*)op);
    }
    else if(this->name == OPN_NN_BATCH_NORM2D) {
        delete((Batch_Norm2d*)op);
    }
    // 量化算子
    else if(this->name == OPN_NN_QCONV2D) {
        delete((QConv2d*)op);
    }
    else if(this->name == OPN_NN_QRELU) {
        delete((QRelu*)op);
    }
    else if(this->name == OPN_QINPUT) {
        delete((QInput*)op);
    }
    else if(this->name == OPN_NN_QMAXPOOL2D) {
        delete((QMaxpool2d*)op);
    }
    else if(this->name == OPN_NN_QFLATTEN) {
        delete((QFlatten*)op);
    }
    else if(this->name == OPN_NN_QDENSE) {
        delete((QDense*)op);
    }
    else if(this->name == OPN_QOUTPUT) {
        delete((QOutput*)op);
    }
    else if(this->name == OPN_QADD) {
        delete((QAdd*)op);
    }
    else if(this->name == OPN_QCONCAT) {
        delete((QConcat*)op);
    }
}

void Node::forward(const std::vector<void *> &intermediate_results, void *input)
{
    /*
     * 前向传播函数
     * 传入存储所有中间结果指针的vector，和graph的input的指针
     * 根据算子名称分类处理：调用算子的forward，传入input和output指针
     */
    // 普通算子
    if(this->name == OPN_NN_CONV2D) {
        ((Conv2d*)op)->forward(
                (Tensor<float32>*)intermediate_results[((Conv2d*)op)->input_node],
                (Tensor<float32>*)intermediate_results[this->number]);
    }
    else if(this->name == OPN_NN_RELU) {
        ((Relu*)op)->forward(
                (Tensor<float32>*)intermediate_results[((Relu*)op)->input_node],
                (Tensor<float32>*)intermediate_results[this->number]);
    }
    else if(this->name == OPN_INPUT) {
        ((Input*)op)->forward(
                (Tensor<float32>*)input,
                (Tensor<float32>*)intermediate_results[this->number]);
    }
    else if(this->name == OPN_NN_MAXPOOL2D) {
        ((Maxpool2d*)op)->forward(
                (Tensor<float32>*)intermediate_results[((Maxpool2d*)op)->input_node],
                (Tensor<float32>*)intermediate_results[this->number]);
    }
    else if(this->name == OPN_NN_FLATTEN) {
        ((Flatten*)op)->forward(
                (Tensor<float32>*)intermediate_results[((Flatten*)op)->input_node],
                (Tensor<float32>*)intermediate_results[this->number]);
    }
    else if(this->name == OPN_NN_DENSE) {
        ((Dense*)op)->forward(
                (Tensor<float32>*)intermediate_results[((Dense*)op)->input_node],
                (Tensor<float32>*)intermediate_results[this->number]);
    }
    else if(this->name == OPN_OUTPUT) {
        ((Output*)op)->forward(
                (Tensor<float32>*)intermediate_results[((Output*)op)->input_node],
                (Tensor<float32>*)intermediate_results[this->number]);
    }
    else if(this->name == OPN_ADD) {
        ((Add*)op)->forward(
                (Tensor<float32>*)intermediate_results[((Add*)op)->input_node1],
                (Tensor<float32>*)intermediate_results[((Add*)op)->input_node2],
                (Tensor<float32>*)intermediate_results[this->number]);
    }
    else if(this->name == OPN_CONCAT) {
        ((Concat*)op)->forward(
                (Tensor<float32>*)intermediate_results[((Concat*)op)->input_node1],
                (Tensor<float32>*)intermediate_results[((Concat*)op)->input_node2],
                (Tensor<float32>*)intermediate_results[this->number]);
    }
    else if(this->name == OPN_NN_BATCH_NORM2D) {
        ((Batch_Norm2d*)op)->forward(
                (Tensor<float32>*)intermediate_results[((Batch_Norm2d*)op)->input_node],
                (Tensor<float32>*)intermediate_results[this->number]);
    }
    // 量化算子
    else if(this->name == OPN_NN_QCONV2D) {
        ((QConv2d*)op)->forward(
                (Tensor<uint8>*)intermediate_results[((QConv2d*)op)->input_node],
                (Tensor<uint8>*)intermediate_results[this->number]);
    }
    else if(this->name == OPN_QINPUT) {
        ((QInput*)op)->forward(
                (Tensor<uint8>*)input,
                (Tensor<uint8>*)intermediate_results[this->number]);
    }
    else if(this->name == OPN_NN_QMAXPOOL2D) {
        ((QMaxpool2d*)op)->forward(
                (Tensor<uint8>*)intermediate_results[((QMaxpool2d*)op)->input_node],
                (Tensor<uint8>*)intermediate_results[this->number]);
    }
    else if(this->name == OPN_NN_QRELU) {
        ((QRelu*)op)->forward(
                (Tensor<uint8>*)intermediate_results[((QRelu*)op)->input_node],
                (Tensor<uint8>*)intermediate_results[this->number]);
    }
    else if(this->name == OPN_NN_QFLATTEN) {
        ((QFlatten*)op)->forward(
                (Tensor<uint8>*)intermediate_results[((QFlatten*)op)->input_node],
                (Tensor<uint8>*)intermediate_results[this->number]);
    }
    else if(this->name == OPN_NN_QDENSE) {
        ((QDense*)op)->forward(
                (Tensor<uint8>*)intermediate_results[((QDense*)op)->input_node],
                (Tensor<uint8>*)intermediate_results[this->number]);
    }
    else if(this->name == OPN_QOUTPUT) {
        ((QOutput*)op)->forward(
                (Tensor<uint8>*)intermediate_results[((QOutput*)op)->input_node],
                (Tensor<uint8>*)intermediate_results[this->number]);
    }
    else if(this->name == OPN_QADD) {
        ((QAdd*)op)->forward(
                (Tensor<uint8>*)intermediate_results[((QAdd*)op)->input_node1],
                (Tensor<uint8>*)intermediate_results[((QAdd*)op)->input_node2],
                (Tensor<uint8>*)intermediate_results[this->number]);
    }
    else if(this->name == OPN_QCONCAT) {
        ((QConcat*)op)->forward(
                (Tensor<uint8>*)intermediate_results[((QConcat*)op)->input_node1],
                (Tensor<uint8>*)intermediate_results[((QConcat*)op)->input_node2],
                (Tensor<uint8>*)intermediate_results[this->number]);
    }
}

void Node::print() {
    /*
     * 打印节点参数
     */
    printf("%%%d=", number);
    // 算子
    if(this->name == OPN_NN_CONV2D) {
        ((Conv2d*)op)->print();
    }
    else if(this->name == OPN_NN_RELU) {
        ((Relu*)op)->print();
    }
    else if(this->name == OPN_INPUT) {
        ((Input*)op)->print();
    }
    else if(this->name == OPN_NN_MAXPOOL2D) {
        ((Maxpool2d*)op)->print();
    }
    else if(this->name == OPN_NN_FLATTEN) {
        ((Flatten*)op)->print();
    }
    else if(this->name == OPN_NN_DENSE) {
        ((Dense*)op)->print();
    }
    else if(this->name == OPN_OUTPUT) {
        ((Output*)op)->print();
    }
    else if(this->name == OPN_ADD) {
        ((Add*)op)->print();
    }
    else if(this->name == OPN_CONCAT) {
        ((Concat*)op)->print();
    }
    else if(this->name == OPN_NN_BATCH_NORM2D) {
        ((Batch_Norm2d*)op)->print();
    }
    // 量化算子
    else if(this->name == OPN_NN_QCONV2D) {
        ((QConv2d*)op)->print();
    }
    else if(this->name == OPN_NN_QRELU) {
        ((QRelu*)op)->print();
    }
    else if(this->name == OPN_QINPUT) {
        ((QInput*)op)->print();
    }
    else if(this->name == OPN_NN_QMAXPOOL2D) {
        ((QMaxpool2d*)op)->print();
    }
    else if(this->name == OPN_NN_QFLATTEN) {
        ((QFlatten*)op)->print();
    }
    else if(this->name == OPN_NN_QDENSE) {
        ((QDense*)op)->print();
    }
    else if(this->name == OPN_QOUTPUT) {
        ((QOutput*)op)->print();
    }
    else if(this->name == OPN_QADD) {
        ((QAdd*)op)->print();
    }
    else if(this->name == OPN_QCONCAT) {
        ((QConcat*)op)->print();
    }
}

Node::Node() {
    /*
     * 创建空node
     */
}

Node *Node::to_qnode() {
    /*
     * 创建量化节点。根据当前节点，创建一个新节点，使其内部的算子为对应的量化算子
     */
    Node *qnode = new Node();
    qnode->number = number;
    qnode->dtype = "uint8";
    qnode->output_shape = output_shape;
    if(this->name == OPN_NN_CONV2D) {
        qnode->name = OPN_NN_QCONV2D;
        qnode->op = new QConv2d((Conv2d*)op);
    }
    else if(this->name == OPN_NN_RELU) {
        qnode->name = OPN_NN_QRELU;
        qnode->op = new QRelu((Relu*)op);
    }
    else if(this->name == OPN_INPUT) {
        qnode->name = OPN_QINPUT;
        qnode->op = new QInput((Input*)op);
    }
    else if(this->name == OPN_NN_MAXPOOL2D) {
        qnode->name = OPN_NN_QMAXPOOL2D;
        qnode->op = new QMaxpool2d((Maxpool2d*)op);
    }
    else if(this->name == OPN_NN_FLATTEN) {
        qnode->name = OPN_NN_QFLATTEN;
        qnode->op = new QFlatten((Flatten*)op);
    }
    else if(this->name == OPN_NN_DENSE) {
        qnode->name = OPN_NN_QDENSE;
        qnode->op = new QDense((Dense*)op);
    }
    else if(this->name == OPN_OUTPUT) {
        qnode->name = OPN_QOUTPUT;
        qnode->op = new QOutput((Output*)op);
    }
    else if(this->name == OPN_ADD) {
        qnode->name = OPN_QADD;
        qnode->op = new QAdd((Add*)op);
    }
    else if(this->name == OPN_CONCAT) {
        qnode->name = OPN_QCONCAT;
        qnode->op = new QConcat((Concat*)op);
    }
    else if(this->name == OPN_NN_BATCH_NORM2D) {
        fprintf(stderr, "File node.cpp, line %d. Found Batch_norm2d in fused graph\n", __LINE__);
        exit(-1);
    }

    return qnode;
}


int get_number(const std::string &graph_line)
{
    /*
     * 从计算图的一行中提取节点编号
     */
    assert(graph_line[0] == '%');
    std::string num;
    int index = 1;
    while(is_digit(graph_line[index])) {
        num.push_back(graph_line[index]);
        index++;
    }
    return (int)strtol(num.c_str(), nullptr, 10);
}

int get_name(const std::string &graph_line) {
    /*
     * 从计算图的一行中提取节点名称
     */
    int index = 0;
    while(graph_line[index] != '=') {
        index++;
    }
    index++;
    std::string name;
    while(graph_line[index] != '(') {
        name.push_back(graph_line[index]);
        index++;
    }

    if(name == "nn.conv2d") {
        return OPN_NN_CONV2D;
    }
    else if(name == "nn.maxpool2d") {
        return OPN_NN_MAXPOOL2D;
    }
    else if(name == "nn.relu") {
        return OPN_NN_RELU;
    }
    else if(name == "input") {
        return OPN_INPUT;
    }
    else if(name == "nn.flatten") {
        return OPN_NN_FLATTEN;
    }
    else if(name == "nn.dense") {
        return OPN_NN_DENSE;
    }
    else if(name == "add") {
        return OPN_ADD;
    }
    else if(name == "concat") {
        return OPN_CONCAT;
    }
    else if(name == "output") {
        return OPN_OUTPUT;
    }
    else if(name == "nn.batch_norm2d") {
        return OPN_NN_BATCH_NORM2D;
    }

    return -1;
}

std::vector<std::string> get_parameters(const std::string &graph_line) {
    /*
     * 提取参数(参数名和参数值 对)。每个参数对存为一个string，并存入vector
     */
    // 从参数名后的括号里取出所有参数组成的字符串
    std::string parameter_line;
    int index = 0;
    while(graph_line[index] != '(') {
        index++;
    }
    index++;
    int bracket = 0;
    while(true) {
        if(graph_line[index] == '(') {
            bracket++;
        }
        if(graph_line[index] == ')') {
            if(bracket == 0) {
                break;
            }
            bracket--;
        }
        parameter_line.push_back(graph_line[index]);
        index++;
    }

    // 将所有参数组成的字符串切分为多个string，每个string包含一个参数对
    std::vector<std::string> parameter_pair_list;
    std::string parameter_pair;
    bracket = 0;
    for(const char &i: parameter_line) {
        if(i == ',' && bracket == 0) {
            parameter_pair_list.push_back(parameter_pair);
            parameter_pair = "";
            continue;
        }
        if(i == '(') {
            bracket++;
        }
        else if(i == ')') {
            bracket--;
        }
        parameter_pair.push_back(i);
    }
    parameter_pair_list.push_back(parameter_pair);
    return parameter_pair_list;
}
