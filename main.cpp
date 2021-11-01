

#include <cstdio>
#include <iostream>
#include <string>


#include "graph.h"
#include "arg_handle.h"
#include "vdarray.h"


int main(int argc, char *argv[]) {
    /////////////// test
    ///////////////////////////

    Graph * graph = nullptr;                                    // calculation graph
    std::string calib_set_path;                                 // calibration set path
    int calib_size[4];                                          // calibration shape
    Vdarray<unsigned char>* calib_set;                          // calibration set
    bool calc_running = false;                                  // whether calc running mean var when fusing op
    int running_size;                                           // size of running mean var
    Vdarray<unsigned char>* calc_running_img;                   // data set to calc running mean var
    Vdarray<float>* running_mean;                               // running mean
    Vdarray<float>* running_var;                                // running var

    for(int i = 1; i<argc; i++) {
        std::string option(argv[i]);    // read an option from argv
        i++;
        if(i >= argc) {
            std::cerr << "Got none value after option " << option << std::endl;
            return -1;
        }
        std::string value(argv[i]);     // the value followed option

        // process option
        if(option == "--graph") {           // create calc graph
            graph = new Graph(value);
        }
        else if(option == "--calib_set") {  // read the path of a txt which contain the path of the calibration set
            calib_set_path = value;
        }
        else if(option == "--calib_size") { // read the input size during calibration
            get_calib_size(calib_size, value);
        }
        else if(option == "--calc_running") {   // read whether calc running when fusing op
            calc_running = string_to_bool(value);
        }
        else if(option == "--running_size") {   // read size of running_mean and running_var
            running_size = get_running_size(value);
        }
        else if(option == "--calc_running_img_list") {  // read the data set used to calculate the running
            calc_running_img = get_calc_running_img(value, calib_size);
        }
        else if(option == "--running_mean_var_binary") {    // read the binary running data
            get_running_mean_var_binary(running_mean, running_var, value, running_size);
        }
        else if(option == "--running_mean_var_txt") {       // read the txt running data
            get_running_mean_var_txt(running_mean, running_var, value, running_size);
        }
        else {
            std::cerr << "option " << option << " not allowed\n";
        }
    }

    calib_set = get_calib_set(calib_set_path, calib_size);  // read calibration set
    // So far，calc graph is generated
    // and calibration set，calibration shape ，bn data set(or bn data) is got

    // TODO: fuse operator


    // TODO: quantization

    // TODO: save quantized model

    return 0;
}
