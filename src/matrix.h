#ifndef MATRIX_H
#define MATRIX_H
#include "yolo_core.h"

// typedef struct matrix{
//    int rows, cols;
//    float **vals;
//} matrix;

typedef struct
{
  int* assignments;
  matrix centers;
} model;

#ifdef __cplusplus
extern "C" {
#endif

model do_kmeans(matrix data, int k);
matrix make_matrix(int rows, int cols);
void free_matrix(matrix m);
void print_matrix(matrix m);

matrix csv_to_matrix(char* filename);
void matrix_to_csv(matrix m);
matrix hold_out_matrix(matrix* m, int n);
void matrix_add_matrix(matrix from, matrix to);
void scale_matrix(matrix m, float scale);
matrix resize_matrix(matrix m, int size);

float* pop_column(matrix* m, int c);

#ifdef __cplusplus
}
#endif
#endif
