#ifdef __cplusplus
extern "C" {
#endif

int cuda_debug_sync = 0;

#ifdef __cplusplus
}
#endif  // __cplusplus

#ifdef GPU

#include "dark_cuda.h"

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "assert.h"
#include "blas.h"
#include "utils.h"

#pragma comment(lib, "cuda.lib")

#ifdef CUDNN
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "cudnn.lib")
#endif  // USE_CMAKE_LIBS
#endif  // CUDNN

#if defined(CUDNN_HALF) && !defined(CUDNN)
#error "If you set CUDNN_HALF=1 then you must set CUDNN=1"
#endif

int CudaGetDeviceCount()
{
  int device_count = 0;
  cudaError_t status = cudaGetDeviceCount(&device_count);
  CHECK_CUDA(status);
  return device_count;
}

void cuda_set_device(int n)
{
  cudaError_t status = cudaSetDevice(n);
  CHECK_CUDA(status);
}

int cuda_get_device()
{
  int n = 0;
  cudaError_t status = cudaGetDevice(&n);
  CHECK_CUDA(status);
  return n;
}

void check_error(cudaError_t status)
{
  cudaError_t status2 = cudaGetLastError();
  if (status != cudaSuccess)
  {
    const char* s = cudaGetErrorString(status);
    char buffer[256];
    printf("\n CUDA Error: %s\n", s);
    snprintf(buffer, 256, "CUDA Error: %s", s);
#ifdef WIN32
    getchar();
#endif
    error(buffer);
  }
  if (status2 != cudaSuccess)
  {
    const char* s = cudaGetErrorString(status2);
    char buffer[256];
    printf("\n CUDA Error Prev: %s\n", s);
    snprintf(buffer, 256, "CUDA Error Prev: %s", s);
#ifdef WIN32
    getchar();
#endif
    error(buffer);
  }
}

void check_error_extended(
    cudaError_t status, const char* file, int line, const char* date_time)
{
  if (status != cudaSuccess)
  {
    printf("CUDA status Error: file: %s() : line: %d : build time: %s \n", file,
        line, date_time);
    check_error(status);
  }
#if defined(DEBUG) || defined(CUDA_DEBUG)
  cuda_debug_sync = 1;
#endif
  if (cuda_debug_sync)
  {
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
      printf(
          "CUDA status = cudaDeviceSynchronize() Error: file: %s() : line: %d "
          ": build time: %s \n",
          file, line, date_time);
  }
  check_error(status);
}

dim3 cuda_gridsize(size_t n)
{
  size_t k = (n - 1) / BLOCK + 1;
  size_t x = k;
  size_t y = 1;
  if (x > 65535)
  {
    x = ceil(sqrt(k));
    y = (n - 1) / (x * BLOCK) + 1;
  }
  // dim3 d = { (unsigned int)x, (unsigned int)y, 1 };
  dim3 d;
  d.x = x;
  d.y = y;
  d.z = 1;
  // printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
  return d;
}

static cudaStream_t
    streamsArray[16];  // cudaStreamSynchronize( get_cuda_stream() );
static int streamInit[16] = {0};

cudaStream_t get_cuda_stream()
{
  int i = cuda_get_device();
  if (!streamInit[i])
  {
    // printf("Create CUDA-stream \n");
    cudaError_t status = cudaStreamCreate(&streamsArray[i]);
    // cudaError_t status = cudaStreamCreateWithFlags(&streamsArray[i],
    // cudaStreamNonBlocking);
    if (status != cudaSuccess)
    {
      printf(" cudaStreamCreate error: %d \n", status);
      const char* s = cudaGetErrorString(status);
      printf("CUDA Error: %s\n", s);
      status = cudaStreamCreateWithFlags(&streamsArray[i], cudaStreamDefault);
      CHECK_CUDA(status);
    }
    streamInit[i] = 1;
  }
  return streamsArray[i];
}

static cudaStream_t
    streamsArray2[16];  // cudaStreamSynchronize( get_cuda_memcpy_stream() );
static int streamInit2[16] = {0};

cudaStream_t get_cuda_memcpy_stream()
{
  int i = cuda_get_device();
  if (!streamInit2[i])
  {
    cudaError_t status = cudaStreamCreate(&streamsArray2[i]);
    // cudaError_t status = cudaStreamCreateWithFlags(&streamsArray2[i],
    // cudaStreamNonBlocking);
    if (status != cudaSuccess)
    {
      printf(" cudaStreamCreate-Memcpy error: %d \n", status);
      const char* s = cudaGetErrorString(status);
      printf("CUDA Error: %s\n", s);
      status = cudaStreamCreateWithFlags(&streamsArray2[i], cudaStreamDefault);
      CHECK_CUDA(status);
    }
    streamInit2[i] = 1;
  }
  return streamsArray2[i];
}

#ifdef CUDNN
cudnnHandle_t cudnn_handle()
{
  static int init[16] = {0};
  static cudnnHandle_t handle[16];
  int i = cuda_get_device();
  if (!init[i])
  {
    cudnnCreate(&handle[i]);
    init[i] = 1;
    cudnnStatus_t status = cudnnSetStream(handle[i], get_cuda_stream());
    CHECK_CUDNN(status);
  }
  return handle[i];
}

void cudnn_check_error(cudnnStatus_t status)
{
#if defined(DEBUG) || defined(CUDA_DEBUG)
  cudaDeviceSynchronize();
#endif
  if (cuda_debug_sync)
  {
    cudaDeviceSynchronize();
  }
  cudnnStatus_t status2 = CUDNN_STATUS_SUCCESS;
#ifdef CUDNN_ERRQUERY_RAWCODE
  cudnnStatus_t status_tmp = cudnnQueryRuntimeError(
      cudnn_handle(), &status2, CUDNN_ERRQUERY_RAWCODE, NULL);
#endif
  if (status != CUDNN_STATUS_SUCCESS)
  {
    const char* s = cudnnGetErrorString(status);
    char buffer[256];
    printf("\n cuDNN Error: %s\n", s);
    snprintf(buffer, 256, "cuDNN Error: %s", s);
#ifdef WIN32
    getchar();
#endif
    error(buffer);
  }
  if (status2 != CUDNN_STATUS_SUCCESS)
  {
    const char* s = cudnnGetErrorString(status2);
    char buffer[256];
    printf("\n cuDNN Error Prev: %s\n", s);
    snprintf(buffer, 256, "cuDNN Error Prev: %s", s);
#ifdef WIN32
    getchar();
#endif
    error(buffer);
  }
}

void cudnn_check_error_extended(
    cudnnStatus_t status, const char* file, int line, const char* date_time)
{
  if (status != CUDNN_STATUS_SUCCESS)
  {
    printf(
        "\n cuDNN status Error in: file: %s() : line: %d : build time: %s \n",
        file, line, date_time);
    cudnn_check_error(status);
  }
#if defined(DEBUG) || defined(CUDA_DEBUG)
  cuda_debug_sync = 1;
#endif
  if (cuda_debug_sync)
  {
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
      printf(
          "\n cudaError_t status = cudaDeviceSynchronize() Error in: file: "
          "%s() : line: %d : build time: %s \n",
          file, line, date_time);
  }
  cudnn_check_error(status);
}
#endif

cublasHandle_t blas_handle()
{
  static int init[16] = {0};
  static cublasHandle_t handle[16];
  int i = cuda_get_device();
  if (!init[i])
  {
    cublasCreate(&handle[i]);
    cublasStatus_t status = cublasSetStream(handle[i], get_cuda_stream());
    CHECK_CUDA((cudaError_t)status);
    init[i] = 1;
  }
  return handle[i];
}

static float** pinned_ptr = NULL;
static size_t pinned_num_of_blocks = 0;
static size_t pinned_index = 0;
static size_t pinned_block_id = 0;
static size_t const pinned_block_size =
    (size_t)1024 * 1024 * 1024 * 1;  // 1 GB block size
static pthread_mutex_t mutex_pinned = PTHREAD_MUTEX_INITIALIZER;

// free CPU-pinned memory
void free_pinned_memory()
{
  if (pinned_ptr)
  {
    for (size_t k = 0; k < pinned_num_of_blocks; ++k)
    {
      cuda_free_host(pinned_ptr[k]);
    }
    free(pinned_ptr);
    pinned_ptr = NULL;
  }
}

// custom CPU-pinned memory allocation
void pre_allocate_pinned_memory(size_t const size)
{
  size_t const num_of_blocks =
      size / pinned_block_size + ((size % pinned_block_size) ? 1 : 0);
  printf("pre_allocate... pinned_ptr = %p \n", pinned_ptr);

  pthread_mutex_lock(&mutex_pinned);
  if (!pinned_ptr)
  {
    pinned_ptr = (float**)calloc(num_of_blocks, sizeof(float*));
    if (!pinned_ptr)
      error("calloc failed in pre_allocate() \n");

    printf(
        "pre_allocate: size = %lu MB, num_of_blocks = %lu, block_size = %lu MB "
        "\n",
        size / (1024 * 1024), num_of_blocks, pinned_block_size / (1024 * 1024));

    for (size_t k = 0; k < num_of_blocks; ++k)
    {
      cudaError_t e = cudaHostAlloc(
          (void**)&pinned_ptr[k], pinned_block_size, cudaHostRegisterMapped);

      if (e != cudaSuccess)
        fprintf(stderr, " Can't pre-allocate CUDA-pinned buffer on CPU-RAM \n");

      CHECK_CUDA(e);

      if (!pinned_ptr[k])
        error("cudaHostAlloc failed\n");
      else
        printf(" Allocated %lu pinned block \n", pinned_block_size);
    }
    pinned_num_of_blocks = num_of_blocks;
  }
  pthread_mutex_unlock(&mutex_pinned);
}

// simple - get pre-allocated pinned memory
float* cuda_make_array_pinned_preallocated(float* x, size_t n)
{
  pthread_mutex_lock(&mutex_pinned);
  float* x_cpu = NULL;
  size_t const memory_step = 512;  // 4096;
  size_t const size = sizeof(float) * n;
  size_t const allocation_size = ((size / memory_step) + 1) * memory_step;

  if (pinned_ptr && pinned_block_id < pinned_num_of_blocks &&
      (allocation_size < pinned_block_size / 2))
  {
    if ((allocation_size + pinned_index) > pinned_block_size)
    {
      float const filled = 100.0f * pinned_index / pinned_block_size;
      printf("\n Pinned block_id = %lu, filled = %f %% \n", pinned_block_id,
          filled);
      pinned_block_id++;
      pinned_index = 0;
    }
    if ((allocation_size + pinned_index) < pinned_block_size &&
        pinned_block_id < pinned_num_of_blocks)
    {
      x_cpu = (float*)((char*)pinned_ptr[pinned_block_id] + pinned_index);
      pinned_index += allocation_size;
    }
    else
    {
      // printf("Pre-allocated pinned memory is over! \n");
    }
  }

  if (!x_cpu)
  {
    if (allocation_size > pinned_block_size / 2)
    {
      printf("Try to allocate new pinned memory, size = %lu MB \n",
          size / (1024 * 1024));
      cudaError_t e =
          cudaHostAlloc((void**)&x_cpu, size, cudaHostRegisterMapped);
      if (e != cudaSuccess)
        fprintf(stderr,
            " Can't allocate CUDA-pinned memory on CPU-RAM (pre-allocated "
            "memory is over too) \n");
      CHECK_CUDA(e);
    }
    else
    {
      printf("Try to allocate new pinned BLOCK, size = %lu MB \n",
          size / (1024 * 1024));
      pinned_num_of_blocks++;
      pinned_block_id = pinned_num_of_blocks - 1;
      pinned_index = 0;
      pinned_ptr =
          (float**)realloc(pinned_ptr, pinned_num_of_blocks * sizeof(float*));
      cudaError_t e = cudaHostAlloc((void**)&pinned_ptr[pinned_block_id],
          pinned_block_size, cudaHostRegisterMapped);
      if (e != cudaSuccess)
        fprintf(stderr, " Can't pre-allocate CUDA-pinned buffer on CPU-RAM \n");
      CHECK_CUDA(e);
      x_cpu = pinned_ptr[pinned_block_id];
    }
  }

  if (x)
  {
    cudaError_t e =
        cudaMemcpyAsync(x_cpu, x, size, cudaMemcpyDefault, get_cuda_stream());
    CHECK_CUDA(e);
  }

  pthread_mutex_unlock(&mutex_pinned);
  return x_cpu;
}

float* cuda_make_array_pinned(float* x, size_t n)
{
  float* x_gpu;
  size_t size = sizeof(float) * n;
  cudaError_t e = cudaHostAlloc((void**)&x_gpu, size, cudaHostRegisterMapped);
  if (e != cudaSuccess)
    fprintf(stderr, " Can't allocate CUDA-pinned memory on CPU-RAM \n");
  CHECK_CUDA(e);
  if (x)
  {
    e = cudaMemcpyAsync(x_gpu, x, size, cudaMemcpyDefault, get_cuda_stream());
    CHECK_CUDA(e);
  }
  if (!x_gpu)
    error("cudaHostAlloc failed\n");
  return x_gpu;
}

float* cuda_make_array(float* x, size_t n)
{
  float* x_gpu;
  size_t size = sizeof(float) * n;
  cudaError_t e = cudaMalloc((void**)&x_gpu, size);
  if (e != cudaSuccess)
    fprintf(stderr, " Try to set subdivisions=64 in your cfg-file. \n");
  CHECK_CUDA(e);
  if (x)
  {
    e = cudaMemcpyAsync(x_gpu, x, size, cudaMemcpyDefault, get_cuda_stream());
    CHECK_CUDA(e);
  }
  if (!x_gpu)
    error("Cuda malloc failed\n");
  return x_gpu;
}

void** cuda_make_array_pointers(void** x, size_t n)
{
  void** x_gpu;
  size_t size = sizeof(void*) * n;
  cudaError_t e = cudaMalloc((void**)&x_gpu, size);
  if (e != cudaSuccess)
    fprintf(stderr, " Try to set subdivisions=64 in your cfg-file. \n");
  CHECK_CUDA(e);
  if (x)
  {
    e = cudaMemcpyAsync(x_gpu, x, size, cudaMemcpyDefault, get_cuda_stream());
    CHECK_CUDA(e);
  }
  if (!x_gpu)
    error("Cuda malloc failed\n");
  return x_gpu;
}

void cuda_random(float* x_gpu, size_t n)
{
  static curandGenerator_t gen[16];
  static int init[16] = {0};
  int i = cuda_get_device();
  if (!init[i])
  {
    curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen[i], time(0));
    init[i] = 1;
  }
  curandGenerateUniform(gen[i], x_gpu, n);
  CHECK_CUDA(cudaPeekAtLastError());
}

float cuda_compare(float* x_gpu, float* x, size_t n, char* s)
{
  float* tmp = (float*)xcalloc(n, sizeof(float));
  cuda_pull_array(x_gpu, tmp, n);
  // int i;
  // for(i = 0; i < n; ++i) printf("%f %f\n", tmp[i], x[i]);
  axpy_cpu(n, -1, x, 1, tmp, 1);
  float err = dot_cpu(n, tmp, 1, tmp, 1);
  printf("Error %s: %f\n", s, sqrt(err / n));
  free(tmp);
  return err;
}

int* cuda_make_int_array(size_t n)
{
  int* x_gpu;
  size_t size = sizeof(int) * n;
  cudaError_t e = cudaMalloc((void**)&x_gpu, size);
  if (e != cudaSuccess)
    fprintf(stderr, " Try to set subdivisions=64 in your cfg-file. \n");
  CHECK_CUDA(e);
  return x_gpu;
}

int* cuda_make_int_array_new_api(int* x, size_t n)
{
  int* x_gpu;
  size_t size = sizeof(int) * n;
  cudaError_t e = cudaMalloc((void**)&x_gpu, size);
  CHECK_CUDA(e);
  if (x)
  {
    cudaError_t e = cudaMemcpyAsync(
        x_gpu, x, size, cudaMemcpyHostToDevice, get_cuda_stream());
    CHECK_CUDA(e);
  }
  if (!x_gpu)
    error("Cuda malloc failed\n");
  return x_gpu;
}

void cuda_free(float* x_gpu)
{
  cudaError_t e = cudaFree(x_gpu);
  CHECK_CUDA(e);
}

void cuda_free_host(float* x_cpu)
{
  cudaError_t e = cudaFreeHost(x_cpu);
  CHECK_CUDA(e);
}

void cuda_push_array(float* x_gpu, float* x, size_t n)
{
  size_t size = sizeof(float) * n;
  cudaError_t e = cudaMemcpyAsync(
      x_gpu, x, size, cudaMemcpyHostToDevice, get_cuda_stream());
  CHECK_CUDA(e);
}

void cuda_pull_array(float* x_gpu, float* x, size_t n)
{
  size_t size = sizeof(float) * n;
  cudaError_t e = cudaMemcpyAsync(
      x, x_gpu, size, cudaMemcpyDeviceToHost, get_cuda_stream());
  CHECK_CUDA(e);
  cudaStreamSynchronize(get_cuda_stream());
}

void cuda_pull_array_async(float* x_gpu, float* x, size_t n)
{
  size_t size = sizeof(float) * n;
  cudaError_t e =
      cudaMemcpyAsync(x, x_gpu, size, cudaMemcpyDefault, get_cuda_stream());
  check_error(e);
}

int get_number_of_blocks(int array_size, int block_size)
{
  return array_size / block_size + ((array_size % block_size > 0) ? 1 : 0);
}

int get_gpu_compute_capability(int i)
{
  typedef struct cudaDeviceProp cudaDeviceProp;
  cudaDeviceProp prop;
  cudaError_t e = cudaGetDeviceProperties(&prop, i);
  CHECK_CUDA(e);
  int cc = prop.major * 100 + prop.minor * 10;  // __CUDA_ARCH__ format
  return cc;
}

void show_cuda_cudnn_info()
{
  int cuda_version = 0, cuda_driver_version = 0, device_count = 0;
  CHECK_CUDA(cudaRuntimeGetVersion(&cuda_version));
  CHECK_CUDA(cudaDriverGetVersion(&cuda_driver_version));
  fprintf(stderr, " CUDA-version: %d (%d)", cuda_version, cuda_driver_version);
  if (cuda_version > cuda_driver_version)
    fprintf(
        stderr, "\n Warning: CUDA-version is higher than Driver-version! \n");
#ifdef CUDNN
  fprintf(
      stderr, ", cuDNN: %d.%d.%d", CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL);
#endif  // CUDNN
#ifdef CUDNN_HALF
  fprintf(stderr, ", CUDNN_HALF=1");
#endif  // CUDNN_HALF
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  fprintf(stderr, ", GPU count: %d ", device_count);
  fprintf(stderr, " \n");
}

#else  // GPU

#include "yolo_core.h"
int cuda_get_device() { return -1; }
void cuda_set_device(int n) {}

#endif  // GPU
