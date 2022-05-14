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

std::string delete_annotation(const std::string &s, const std::string &annotation) {
    /*
     * 根据输入的注释标记符号，删除注释部分
     */
    unsigned long index = s.find(annotation);
    std::string result;
    if(index != std::string::npos) {
        for (int i = 0; i < (int) index; i++) {
            result.push_back(s[i]);
        }
    }
    else {
        result = s;
    }
    return result;
}

System_info::System_info() {
    n_proc = sysconf(_SC_NPROCESSORS_ONLN);
}

void clear_log()
{
    int ret = system("rm logs/*");
    if(ret == -2147483647) {
        fprintf(stderr, "file util.cpp line %d: Error return value %d\n", __LINE__, ret);
    }
}

void xxlog(const std::string &msg, const std::string &type)
{
    if(access("logs", 0) < 0) {
        std::string cmd = "mkdir logs";
        int ret = system(cmd.c_str());
        if(ret != 0) {
            fprintf(stderr, "file util.cpp line %d: Error return value %d\n", __LINE__, ret);
        }
    }

    time_t t;
    struct tm * lt;
    //获取Unix时间戳。
    time (&t);
    //转为时间结构。
    lt = localtime (&t);

    FILE * file = fopen("logs/quant_log.log", "a");
    fprintf(file, "[%d-%d-%d %d:%d:%d] %s: %s\n", 
        lt->tm_year+1990, lt->tm_mon, lt->tm_mday, lt->tm_hour, lt->tm_min, lt->tm_sec, type.c_str(), msg.c_str());
    fclose(file);
}