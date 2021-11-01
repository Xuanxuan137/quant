//
// Created by noname on 2021/10/28.
//

#include "node.h"

Node::Node(const std::string& read_graph_line,
           const std::vector<std::vector<int> > &output_shape_list) {
    /*
     * Constructor:
     * According to the input calculation graph operator information graph_line,
     * 1. Extract the number and set the value for the 'number' attribute
     * 2. Extract the operator name and set the value for the 'name' attribute
     * 3. According to the operator name, set the value for the dtype attribute
     * 4. Create an operator object according to the operator type, and pass
     *      in graph_line to initialize the attributes of the operator
     * 5. According to the output_shape attribute of the operator, set the value
     *      for the output_shape attribute of the node
     */
    std::cout << read_graph_line << std::endl;
    std::string graph_line = replace(read_graph_line, " ", "");

    // Extract serial number
    this->number = get_number(graph_line);
    // Extract operator name
    this->name = get_name(graph_line);
    // Extract operator parameters. Separate all parameters, save each as a string, and store it in a vector
    std::vector<std::string> parameters = get_parameters(graph_line);
    // Create an op object based on the operator name
    // Is there a better way to achieve it? Don’t need a long list of if else? I can't think of
    if(this->name == "nn.conv2d") {
        this->dtype = "float32";
        op = new Conv2d(parameters, output_shape_list);
        this->output_shape = ((Conv2d*)op)->output_shape;
    }
    else if(this->name == "nn.relu") {
        this->dtype = "float32";
        op = new Relu(parameters, output_shape_list);
        this->output_shape = ((Relu*)op)->output_shape;
    }
    else if(this->name == "nn.maxpool2d") {
        this->dtype = "float32";
        op = new Maxpool2d(parameters, output_shape_list);
        this->output_shape = ((Maxpool2d*)op)->output_shape;
    }
    else if(this->name == "input") {
        this->dtype = "float32";
        op = new Input(parameters);
        this->output_shape = ((Input*)op)->output_shape;
    }
    else if(this->name == "nn.flatten") {
        this->dtype = "float32";
        op = new Flatten(parameters, output_shape_list);
        this->output_shape = ((Flatten*)op)->output_shape;
    }
    else if(this->name == "nn.dense") {
        this->dtype = "float32";
        op = new Dense(parameters, output_shape_list);
        this->output_shape = ((Dense*)op)->output_shape;
    }
//    else if(this->name == "output") {
//        this->dtype = "float32";
//        op = new Output(parameters, output_shape_list);
//        this->output_shape = ((Output*)op)->output_shape;
//    }
//    else if(this->name == "add") {
//        this->dtype = "float32";
//        op = new Add(parameters, output_shape_list);
//        this->output_shape = ((Add*)op)->output_shape;
//    }
//    else if(this->name == "concat") {
//        this->dtype = "float32";
//        op = new Concat(parameters, output_shape_list);
//        this->output_shape = ((Concat*)op)->output_shape;
//    }
}

Node::~Node() {
    /*
     * destructor: free op
     */
    if(0) {}
//    else if(this->name == "nn.conv2d") {
//        delete((Conv2d*)op);
//    }
    else if(this->name == "nn.relu") {
        printf("delete relu:\n");
        delete((Relu*)op);
    }
    else if(this->name == "input") {
        printf("delete input:\n");
        delete((Input*)op);
    }
    else if(this->name == "nn.maxpool2d") {
        printf("delete maxpool2d:\n");
        delete((Maxpool2d*)op);
    }
//    else if(this->name == "nn.flatten") {
//        delete((Flatten*)op);
//    }
//    else if(this->name == "nn.dense") {
//        delete((Dense*)op);
//    }
//    else if(this->name == "output") {
//        delete((Output*)op);
//    }
//    else if(this->name == "add") {
//        delete((Add*)op);
//    }
//    else if(this->name == "concat") {
//        delete((Concat*)op);
//    }
}


int get_number(const std::string &graph_line)
{
    /*
     * Extract the node number from one line of the input calculation graph
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

std::string get_name(const std::string &graph_line) {
    /*
     * Extract the name of the node operator from a line of the input calculation graph
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
    return name;
}

std::vector<std::string> get_parameters(const std::string &graph_line) {
    /*
     * Extract the parameters (parameter name and parameter value pair) of the operator from one line of the
     * input calculation graph. Each parameter pair is saved as a string, and these parameter strings are
     * added to the vector
     */
    // Take out the parameter string from the parentheses after the operator name in a line of the input
    // calculation graph. All parameters are in this string
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

    // Split the parameter string into multiple strings, each string contains a parameter pair
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
