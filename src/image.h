#pragma once
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "box.h"
#include "libapi.h"

typedef enum
{
  PNG,
  BMP,
  TGA,
  JPG
} IMTYPE;

typedef struct Image
{
  int w;
  int h;
  int c;
  float* data;
} Image;

LIB_API void make_image_red(Image im);
LIB_API Image make_attention_image(int img_size, float* original_delta_cpu,
    float* original_input_cpu, int w, int h, int c);
LIB_API Image resize_image(Image im, int w, int h);
LIB_API Image make_image(int w, int h, int c);
LIB_API Image load_image_color(char* filename, int w, int h);
LIB_API void free_image(Image m);
LIB_API Image crop_image(Image im, int dx, int dy, int w, int h);

void flip_image(Image a);
void draw_box(
    Image a, int x1, int y1, int x2, int y2, float r, float g, float b);
void draw_box_width(
    Image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);
void scale_image(Image m, float s);
Image random_crop_image(Image im, int w, int h);
Image random_augment_image(
    Image im, float angle, float aspect, int low, int high, int size);
void random_distort_image(
    Image im, float hue, float saturation, float exposure);
void fill_image(Image m, float s);
void embed_image(Image source, Image dest, int dx, int dy);
void distort_image(Image im, float hue, float sat, float val);
void hsv_to_rgb(Image im);
void constrain_image(Image im);

void show_image(Image p, const char* name);
void save_image(Image p, const char* name);

Image make_empty_image(int w, int h, int c);
Image copy_image(Image p);
Image load_image(char const* filename, int w, int h, int c);
Image load_image_stb_resize(char* filename, int w, int h, int c);

float bilinear_interpolate(Image im, float x, float y, int c);
