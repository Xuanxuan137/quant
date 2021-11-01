//
// Created by noname on 2021/10/23.
//

#include "graph.h"


Graph::Graph(const std::string& path)
{
    /*
     * read file in path, create calculation graph with file
     * open file, create a node with each line
     */
    printf("Reading calculation graph...\n");

    // open calculation graph file
    std::ifstream graph_file;
    graph_file.open(path, std::ios::in);
    if(!graph_file.is_open()) {
        std::cerr << "calib set txt file not found\n";
        exit(-1);
    }
    // create a node with each line
    std::string graph_line;
    while(std::getline(graph_file, graph_line)) {
        if(replace(graph_line, " ", "") == "\n") {
            continue;
        }
        Node new_node(graph_line, output_shape_list);
        printf("line: %d\n", __LINE__);
        printf("%p\n", &new_node);
        node_list.push_back(new_node);
        printf("line: %d\n", __LINE__);
        output_shape_list.push_back(new_node.output_shape);
    }
}