//
// Created by noname on 2021/10/24.
//

#ifndef QUANT_UTIL_H
#define QUANT_UTIL_H

#include <iostream>
#include <string>
#include <vector>
#include <cstddef>
#include <sys/time.h>
#include <unistd.h>


class System_info {
public:
    int n_proc;         // 可用cpu数量

    System_info();
};


bool string_to_bool(const std::string &value);
std::vector<std::string> split(const std::string &s, const std::string &seperator);
std::string replace(const std::string &s, const std::string &origin, const std::string &substitute);
bool is_digit(char c);
unsigned long long get_micro_sec_time();
std::string delete_annotation(const std::string &s, const std::string &annotation);
#endif //QUANT_UTIL_H
