#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>

#include "darkunistd.h"

#ifndef WIN32
#include <sys/time.h>
#endif

#ifndef USE_CMAKE_LIBS
#pragma warning(disable : 4996)
#endif

void free_ptrs(void** ptrs, int n)
{
  for (int i = 0; i < n; ++i)
  {
    free(ptrs[i]);
  }
  free(ptrs);
}

void* xmalloc(size_t size)
{
  void* ptr = malloc(size);
  if (!ptr)
  {
    printf("xmalloc error\n");
    exit(EXIT_FAILURE);
  }
  return ptr;
}

void* xcalloc(size_t nmemb, size_t size)
{
  void* ptr = calloc(nmemb, size);
  if (!ptr)
  {
    printf("xcalloc error\n");
    exit(EXIT_FAILURE);
  }
  return ptr;
}

void* xrealloc(void* ptr, size_t size)
{
  ptr = realloc(ptr, size);
  if (!ptr)
  {
    printf("xrealloc error\n");
    exit(EXIT_FAILURE);
  }
  return ptr;
}

int* read_map(char* filename)
{
  int n = 0;
  int* map = 0;
  char* str;
  FILE* file = fopen(filename, "r");
  if (!file)
    FileError(filename);
  while ((str = fgetl(file)))
  {
    ++n;
    map = (int*)xrealloc(map, n * sizeof(int));
    map[n - 1] = atoi(str);
    free(str);
  }
  if (file)
    fclose(file);
  return map;
}

char* BaseCfg(char const* cfg_file)
{
  char* c = (char*)cfg_file;
  char* next;
  while ((next = strchr(c, '/')))
  {
    c = next + 1;
  }
  if (!next)
    while ((next = strchr(c, '\\')))
    {
      c = next + 1;
    }
  c = copy_string(c);
  next = strchr(c, '.');
  if (next)
    *next = 0;
  return c;
}

std::string ReplaceImage2Label(std::string str)
{
  size_t idx = str.find_last_of('.');
  str.replace(str.begin() + idx, str.end(), ".txt");

  return str;
}

void error(const char* s)
{
  perror(s);
  assert(0);
  exit(EXIT_FAILURE);
}

void FileError(char const* s)
{
  printf("Couldn't open file: %s\n", s);
  exit(EXIT_FAILURE);
}

void strip(char* s)
{
  size_t i;
  size_t len = strlen(s);
  size_t offset = 0;
  for (i = 0; i < len; ++i)
  {
    char c = s[i];
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0d ||
        c == 0x0a)
      ++offset;
    else
      s[i - offset] = c;
  }
  s[len - offset] = '\0';
}

char* fgetl(FILE* fp)
{
  if (feof(fp))
    return 0;
  size_t size = 512;
  char* line = (char*)xmalloc(size * sizeof(char));
  if (!fgets(line, size, fp))
  {
    free(line);
    return 0;
  }

  size_t curr = strlen(line);

  while ((line[curr - 1] != '\n') && !feof(fp))
  {
    if (curr == size - 1)
    {
      size *= 2;
      line = (char*)xrealloc(line, size * sizeof(char));
    }
    size_t readsize = size - curr;
    if (readsize > INT_MAX)
      readsize = INT_MAX - 1;
    fgets(&line[curr], readsize, fp);
    curr = strlen(line);
  }
  if (curr >= 2)
    if (line[curr - 2] == 0x0d)
      line[curr - 2] = 0x00;

  if (curr >= 1)
    if (line[curr - 1] == 0x0a)
      line[curr - 1] = 0x00;

  return line;
}

char* copy_string(char* s)
{
  if (!s)
  {
    return NULL;
  }
  char* copy = (char*)xmalloc(strlen(s) + 1);
  strncpy(copy, s, strlen(s) + 1);
  return copy;
}

int count_fields(char* line)
{
  int count = 0;
  int done = 0;
  char* c;
  for (c = line; !done; ++c)
  {
    done = (*c == '\0');
    if (*c == ',' || done)
      ++count;
  }
  return count;
}

float* parse_fields(char* line, int n)
{
  float* field = (float*)xcalloc(n, sizeof(float));
  char *c, *p, *end;
  int count = 0;
  int done = 0;
  for (c = line, p = line; !done; ++c)
  {
    done = (*c == '\0');
    if (*c == ',' || done)
    {
      *c = '\0';
      field[count] = strtod(p, &end);
      if (p == c)
        field[count] = nan("");
      if (end != c && (end != c - 1 || *end != '\r'))
        field[count] = nan("");  // DOS file formats!
      p = c + 1;
      ++count;
    }
  }
  return field;
}

float sum_array(float* a, int n)
{
  int i;
  float sum = 0;
  for (i = 0; i < n; ++i) sum += a[i];
  return sum;
}

int constrain_int(int a, int min, int max)
{
  if (a < min)
    return min;
  if (a > max)
    return max;
  return a;
}

float constrain(float min, float max, float a)
{
  if (a < min)
    return min;
  if (a > max)
    return max;
  return a;
}

float dist_array(float* a, float* b, int n, int sub)
{
  int i;
  float sum = 0;
  for (i = 0; i < n; i += sub) sum += pow(a[i] - b[i], 2);
  return sqrt(sum);
}

float mag_array(float* a, int n)
{
  int i;
  float sum = 0;
  for (i = 0; i < n; ++i)
  {
    sum += a[i] * a[i];
  }
  return sqrt(sum);
}

int int_index(int* a, int val, int n)
{
  int i;
  for (i = 0; i < n; ++i)
  {
    if (a[i] == val)
      return i;
  }
  return -1;
}

unsigned int RandGen()
{
  unsigned int rnd = 0;
#ifdef WIN32
  rand_s(&rnd);
#else  // WIN32
  rnd = rand();
#if (RAND_MAX < 65536)
  rnd = rand() * (RAND_MAX + 1) + rnd;
#endif  //(RAND_MAX < 65536)
#endif  // WIN32
  return rnd;
}

int RandInt(int min, int max)
{
  if (max < min)
  {
    int s = min;
    min = max;
    max = s;
  }
  int r = (RandGen() % (max - min + 1)) + min;
  return r;
}

float RandFloat()
{
  unsigned int rnd = 0;
#ifdef WIN32
  rand_s(&rnd);
  return ((float)rnd / (float)UINT_MAX);
#else  // WIN32

  rnd = rand();
#if (RAND_MAX < 65536)
  rnd = rand() * (RAND_MAX + 1) + rnd;
  return ((float)rnd / (float)(RAND_MAX * RAND_MAX));
#endif  //(RAND_MAX < 65536)
  return ((float)rnd / (float)RAND_MAX);

#endif  // WIN32
}

// From http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
float RandNormal()
{
  static int haveSpare = 0;
  static double rand1, rand2;

  if (haveSpare)
  {
    haveSpare = 0;
    return sqrt(rand1) * sin(rand2);
  }

  haveSpare = 1;

  rand1 = RandGen() / ((double)RAND_MAX);
  if (rand1 < 1e-100)
    rand1 = 1e-100;
  rand1 = -2 * log(rand1);
  rand2 = (RandGen() / ((double)RAND_MAX)) * 2.0 * M_PI;

  return sqrt(rand1) * cos(rand2);
}

float RandScale(float s)
{
  float scale = RandUniformStrong(1, s);
  if (RandGen() % 2)
    return scale;
  return 1. / scale;
}

float RandUniform(float min, float max)
{
  if (max < min)
  {
    float swap = min;
    min = max;
    max = swap;
  }

#if (RAND_MAX < 65536)
  int rnd = rand() * (RAND_MAX + 1) + rand();
  return ((float)rnd / (RAND_MAX * RAND_MAX) * (max - min)) + min;
#else
  return ((float)rand() / RAND_MAX * (max - min)) + min;
#endif
}

float RandUniformStrong(float min, float max)
{
  if (max < min)
  {
    float swap = min;
    min = max;
    max = swap;
  }
  return (RandFloat() * (max - min)) + min;
}

float RandPreCalc(float min, float max, float random_part)
{
  if (max < min)
  {
    float swap = min;
    min = max;
    max = swap;
  }
  return (random_part * (max - min)) + min;
}

int MakeDir(char const* path, int mode)
{
#ifdef WIN32
  return _mkdir(path);
#else
  return mkdir(path, mode);
#endif
}

bool Exists(char const* path)
{
  struct stat info;
  if (stat(path, &info) != 0)
    return false;
  else
    return true;
}

#if __cplusplus >= 201103L || _MSC_VER >= 1900  // C++11
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

// timer related functions
float Clocks2Sec(clock_t clocks) { return (float)clocks / CLOCKS_PER_SEC; }

static std::chrono::steady_clock::time_point steady_start, steady_end;
static double total_time = 0.0;

double GetTimePoint()
{
  std::chrono::steady_clock::time_point current_time =
      std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(
      current_time.time_since_epoch())
      .count();
}

void StartGlobalTimer() { steady_start = std::chrono::steady_clock::now(); }

void StopGlobalTimer() { steady_end = std::chrono::steady_clock::now(); }

double GetGlobalTime()
{
  double took_time =
      std::chrono::duration<double>(steady_end - steady_start).count();
  total_time += took_time;
  return took_time;
}

void StopGlobalTimerAndShow(char* name)
{
  StopGlobalTimer();
  if (name != nullptr)
    std::cout << name << ": ";
  std::cout << GetGlobalTime() * 1000 << " ms" << std::endl;
}

void ShowGlobalTotalTime()
{
  std::cout << " Total: " << total_time * 1000 << " msec" << std::endl;
}

// thread related functions
int custom_atomic_load_int(volatile int* obj)
{
  const volatile std::atomic<int>* ptr_a =
      (const volatile std::atomic<int>*)obj;
  return std::atomic_load(ptr_a);
}

void custom_atomic_store_int(volatile int* obj, int desr)
{
  volatile std::atomic<int>* ptr_a = (volatile std::atomic<int>*)obj;
  std::atomic_store(ptr_a, desr);
}

void this_thread_sleep_for(int ms_time)
{
  std::chrono::milliseconds dura(ms_time);
  std::this_thread::sleep_for(dura);
}

void this_thread_yield() { std::this_thread::yield(); }
#endif  // C++11