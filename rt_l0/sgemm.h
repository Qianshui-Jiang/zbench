#pragma once

#include <fstream>

int run_sgemm(int m, int niterations, int gx, int gy, 
            const char* bin_file, const char* fn_name);

int run_bgemm(int M, int K, int N, int threadWidth, int threadHeight,
            int groupWidth, int groupHeight,
            const char* bin_file, const char* fn_name);