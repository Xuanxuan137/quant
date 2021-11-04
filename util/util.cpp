//
// Created by noname on 2021/10/24.
//

#include "util.h"

bool string_to_bool(const std::string &value)
{
    if(value == "true" || value == "True" || value == "TRUE") {
        return true;
    }
    else if(value == "false" || value == "False" || value == "FALSE") {
        return false;
    }
    else {
        std::cerr << "Wrong bool value\n";
        exit(-1);
    }
}

std::vector<std::string> split(const std::string &s, const std::string &seperator)
{
    std::vector<std::string> target;
    unsigned long s_len = s.size();
    unsigned long seperator_len = seperator.size();

    unsigned long p = 0;
    while(true) {
        unsigned long index = s.find(seperator, p);
        if(index != std::string::npos) {
            if(index-p != 0) {
                target.push_back(s.substr(p, index-p));
            }
            p = index + seperator_len;
        }
        else {
            if(s_len-p != 0) {
                target.push_back(s.substr(p, s_len - p));
            }
            break;
        }
    }
    return target;
}

std::string replace(const std::string &s, const std::string &origin, const std::string &substitute)
{
    std::string target;
    unsigned long s_len = s.size();
    unsigned long origin_len = origin.size();

    unsigned long p = 0;
    while(true) {
        unsigned long index = s.find(origin, p);
        if(index != std::string::npos) {
            target += s.substr(p, index-p);
            target += substitute;
            p = index + origin_len;
        }
        else {
            target += s.substr(p, s_len-p);
            break;
        }
    }
    return target;
}

bool is_digit(char c) {
    return (c >= '0' && c <= '9');
}


unsigned long long get_micro_sec_time() {
    struct timeval time_val;
    gettimeofday(&time_val, NULL);
    return time_val.tv_sec * 1000000 + time_val.tv_usec;
}