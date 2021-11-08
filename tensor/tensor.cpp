//
// Created by noname on 2021/10/23.
//

#include "tensor.h"

void array_add_1(int array[], const std::vector<int> &size)
{
    /*
     * 给出一个数组array，和数组每一位的进制size。让数组自增一
     * 常用于Tensor按下标遍历。调用此函数可将下标加一
     */
    int dim = (int)size.size();
    int p = dim-1;
    array[p]++;
    while(true) {
        if(array[p] >= size[p]) {
            array[p] = 0;
            p--;
            if(p < 0) {
                break;
            }
            array[p]++;
        }
        else {
            break;
        }
    }
}

void print_size(const std::vector<int> &size)
{
    /*
     * 打印一个Tensor的size
     */
    for(const int &i: size) {
        printf("%d ", i);
    }
    printf("\n");
}

void mt_dot(float * C, float * A, float * B, int mt_M, int mt_K, int mt_N)
{
    /*
     * 多线程矩阵乘法的一个子线程
     */
    int new_M = mt_M / 4 * 4;
    for (int i = 0; i < new_M; i+=4) {
        int new_N = mt_N / 4 * 4;
        for (int j = 0; j < new_N; j += 4) {
            float temp11 = 0; float temp12 = 0; float temp13 = 0; float temp14 = 0;
            float temp21 = 0; float temp22 = 0; float temp23 = 0; float temp24 = 0;
            float temp31 = 0; float temp32 = 0; float temp33 = 0; float temp34 = 0;
            float temp41 = 0; float temp42 = 0; float temp43 = 0; float temp44 = 0;
            for (int k = 0; k < mt_K; k++) {
                temp11 += A[i * mt_K + k] * B[k * mt_N + j];
                temp12 += A[(i+1) * mt_K + k] * B[k * mt_N + j];
                temp13 += A[(i+2) * mt_K + k] * B[k * mt_N + j];
                temp14 += A[(i+3) * mt_K + k] * B[k * mt_N + j];
                temp21 += A[i * mt_K + k] * B[k * mt_N + j + 1];
                temp22 += A[(i+1) * mt_K + k] * B[k * mt_N + j + 1];
                temp23 += A[(i+2) * mt_K + k] * B[k * mt_N + j + 1];
                temp24 += A[(i+3) * mt_K + k] * B[k * mt_N + j + 1];
                temp31 += A[i * mt_K + k] * B[k * mt_N + j + 2];
                temp32 += A[(i+1) * mt_K + k] * B[k * mt_N + j + 2];
                temp33 += A[(i+2) * mt_K + k] * B[k * mt_N + j + 2];
                temp34 += A[(i+3) * mt_K + k] * B[k * mt_N + j + 2];
                temp41 += A[i * mt_K + k] * B[k * mt_N + j + 3];
                temp42 += A[(i+1) * mt_K + k] * B[k * mt_N + j + 3];
                temp43 += A[(i+2) * mt_K + k] * B[k * mt_N + j + 3];
                temp44 += A[(i+3) * mt_K + k] * B[k * mt_N + j + 3];
            }
            C[i * mt_N + j] = temp11;
            C[(i+1) * mt_N + j] = temp12;
            C[(i+2) * mt_N + j] = temp13;
            C[(i+3) * mt_N + j] = temp14;
            C[i * mt_N + j + 1] = temp21;
            C[(i+1) * mt_N + j + 1] = temp22;
            C[(i+2) * mt_N + j + 1] = temp23;
            C[(i+3) * mt_N + j + 1] = temp24;
            C[i * mt_N + j + 2] = temp31;
            C[(i+1) * mt_N + j + 2] = temp32;
            C[(i+2) * mt_N + j + 2] = temp33;
            C[(i+3) * mt_N + j + 2] = temp34;
            C[i * mt_N + j + 3] = temp41;
            C[(i+1) * mt_N + j + 3] = temp42;
            C[(i+2) * mt_N + j + 3] = temp43;
            C[(i+3) * mt_N + j + 3] = temp44;
        }
        for (int j = new_N; j < mt_N; j++) {
            float temp1 = 0; float temp2 = 0; float temp3 = 0; float temp4 = 0;
            for (int k = 0; k < mt_K; k++) {
                temp1 += A[i * mt_K + k] * B[k * mt_N + j];
                temp2 += A[(i+1) * mt_K + k] * B[k * mt_N + j];
                temp3 += A[(i+2) * mt_K + k] * B[k * mt_N + j];
                temp4 += A[(i+3) * mt_K + k] * B[k * mt_N + j];
            }
            C[i * mt_N + j] = temp1;
            C[(i+1) * mt_N + j] = temp2;
            C[(i+2) * mt_N + j] = temp3;
            C[(i+3) * mt_N + j] = temp4;
        }
    }
    for (int i = new_M; i < mt_M; i++) {
        int new_N = mt_N / 4 * 4;
        for (int j = 0; j < new_N; j += 4) {
            float temp1 = 0;
            float temp2 = 0;
            float temp3 = 0;
            float temp4 = 0;
            for (int k = 0; k < mt_K; k++) {
                temp1 += A[i * mt_K + k] * B[k * mt_N + j];
                temp2 += A[i * mt_K + k] * B[k * mt_N + j + 1];
                temp3 += A[i * mt_K + k] * B[k * mt_N + j + 2];
                temp4 += A[i * mt_K + k] * B[k * mt_N + j + 3];
            }
            C[i * mt_N + j] = temp1;
            C[i * mt_N + j + 1] = temp2;
            C[i * mt_N + j + 2] = temp3;
            C[i * mt_N + j + 3] = temp4;
        }
        for (int j = new_N; j < mt_N; j++) {
            float temp = 0;
            for (int k = 0; k < mt_K; k++) {
                temp += A[i * mt_K + k] * B[k * mt_N + j];
            }
            C[i * mt_N + j] = temp;
        }
    }
}