//
// Created by noname on 2021/10/24.
//

#ifndef QUANT_UTIL_H
#define QUANT_UTIL_H

#include <iostream>
#include <string>
#include <vector>


bool string_to_bool(const std::string &value);
std::vector<std::string> split(const std::string &s, const std::string &seperator);
std::string replace(const std::string &s, const std::string &origin, const std::string &substitute);
bool is_digit(char c);

#endif //QUANT_UTIL_H
