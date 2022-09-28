#include <iostream>
#include "cmath"
#include "omp.h"
#include "chrono"

using namespace std;
FILE* file;

void writeAns(unsigned int* ans, int len, char* out) {
    file = fopen(out, "wb");
    fwrite(ans, sizeof(unsigned int), len, file);
    fclose(file);
}


unsigned int* calculateGistWithoutOMP(int colors, int len, int* bitmap) {
    unsigned int* gist = new unsigned int[colors] {};
    for (int pos = 0; pos < len; pos++) gist[bitmap[pos]]++;
    return gist;
}


unsigned int* calculateGist(int colors, int len, int* bitmap, int threads) {
    unsigned int* ans = new unsigned int[colors] {};
    int n = threads;
#pragma omp parallel for num_threads(threads) schedule(static)

    for (int thread = 0; thread < n; thread++) {
        unsigned int* gist = new unsigned int[colors] {};
        for (int i = thread; i < len; i += n) {
            gist[bitmap[i]]++;
        }
#pragma omp critical
        for (int i = 0; i < colors; i++) {
            ans[i] += gist[i];
        }
    }
    return ans;
}


unsigned int* calculateGist_2(int colors, int len, int* bitmap, int threads) {
    unsigned int* ans = new unsigned int[colors] {};
#pragma omp parallel num_threads(threads)
    {
        unsigned int* gist = new unsigned int[colors] {};
#pragma omp for schedule(static)
        for (int i = 0; i < len; i++) {
            gist[bitmap[i]]++;
        }
#pragma omp critical
        for (int i = 0; i < colors; i++) {
            ans[i] += gist[i];
        }
    }
    return ans;
}


unsigned int* calculateGist_3(int colors, int len, int* bitmap, int threads) {
    unsigned int* ans = new unsigned int[colors*threads] {};
#pragma omp parallel num_threads(threads)
    {
        unsigned int* pointer = &ans[omp_get_thread_num() * colors];
#pragma omp for schedule(static)
        for (int i = 0; i < len; i++) {
            ++pointer[bitmap[i]];
        }
    }
    unsigned int* zero_pointer = &ans[0];
    unsigned int* pointer;
    for (int i = 1; i < threads; i++) {
        pointer = &ans[colors*i];
        for (int j = 0; j < colors; j++) {
            zero_pointer[j] += pointer[j];
        }
    }
    return ans;
}

int main(int argc, char** argv) {

    char* input_file_name = argv[1];
    char* output_file_name = argv[2];
    int num_threads = atoi(argv[3]);

    file = fopen(input_file_name, "rb");
    if (!file) return 1;
    getc(file);
    getc(file);
    int height = 0, width = 0;
    fscanf(file, "%d %d\n", &width, &height);
    int colors = 0;
    fscanf(file, "%d\n", &colors);
    colors++;
    int len = height * width;
    int* bitmap = new int[len]{};

    for (int i = 0; i < len; i++) {
        bitmap[i] = int(getc(file));
    }
    fclose(file);

    int attempts = 100;
    double time_ms;
    double all_time = 0;
    unsigned int* gist;
    for (int i = 0; i < attempts; i++){
        if (num_threads == -1) {
            auto start_time = chrono::steady_clock::now();
            gist = calculateGistWithoutOMP(colors, len, bitmap);
            auto end_time = chrono::steady_clock::now();
            time_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / 10e6;
        }
        else {
            if (num_threads == 0) {
                num_threads = omp_get_max_threads();
            }
            auto start_time = chrono::steady_clock::now();
            gist = calculateGist_3(colors, len, bitmap, num_threads);
            auto end_time = chrono::steady_clock::now();
            time_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / 10e6;
        }
        all_time+=time_ms;
    }
    double average_time = all_time / attempts;
    writeAns(gist, colors, output_file_name);
    printf("Time (%i thread(s)): %g ms\n", num_threads, average_time);
    return 0;
}