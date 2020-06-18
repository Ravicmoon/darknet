#pragma once

#include <stdio.h>
#include <time.h>

#include "yolo_core.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define max_val_cmp(a, b) (((a) > (b)) ? (a) : (b))
#define min_val_cmp(a, b) (((a) < (b)) ? (a) : (b))

void* xmalloc(size_t size);
void* xcalloc(size_t nmemb, size_t size);
void* xrealloc(void* ptr, size_t size);

int* read_map(char* filename);
char* BaseCfg(char const* cfgfile);
std::string ReplaceImage2Label(std::string str);
void error(const char* s);
void FileError(char const* s);
void strip(char* s);
char* fgetl(FILE* fp);
char* copy_string(char* s);
int count_fields(char* line);
float* parse_fields(char* line, int n);
float constrain(float min, float max, float a);
int constrain_int(int a, int min, int max);

float sum_array(float* a, int n);
float mag_array(float* a, int n);
float dist_array(float* a, float* b, int n, int sub);

unsigned int random_gen();
int rand_int(int min, int max);
float random_float();
float rand_normal();
float rand_uniform(float min, float max);
float RandScale(float s);
float rand_uniform_strong(float min, float max);
float rand_precalc_random(float min, float max, float random_part);

int int_index(int* a, int val, int n);
int MakeDir(char const* path, int mode);
bool Exists(char const* path);

#if __cplusplus >= 201103L || _MSC_VER >= 1900  // C++11
// timer related functions
float Clocks2Sec(clock_t clocks);
double GetTimePoint();
void StartGlobalTimer();
void StopGlobalTimer();
double GetGlobalTime();
void StopGlobalTimerAndShow(char* name);
void ShowGlobalTotalTime();

// thread related functions
int custom_atomic_load_int(volatile int* obj);
void custom_atomic_store_int(volatile int* obj, int desr);
void this_thread_sleep_for(int ms_time);
void this_thread_yield();
#endif  // C++11