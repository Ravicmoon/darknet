#pragma once

#include "image.h"
#include "matrix.h"

// declaration
typedef void* mat_cv;
typedef void* cap_cv;
typedef void* write_cv;

mat_cv* load_image_mat_cv(const char* filename, int flag);
Image load_image_cv(char const* filename, int channels);
Image load_image_resize(char* filename, int w, int h, int c, Image* im);
int get_width_mat(mat_cv* mat);
int get_height_mat(mat_cv* mat);
void release_mat(mat_cv** mat);

// Window
void create_window_cv(
    char const* window_name, int full_screen, int width, int height);
void destroy_all_windows_cv();
int wait_key_cv(int delay);
int wait_until_press_key_cv();
void make_window(char* name, int w, int h, int fullscreen);
void show_image_cv(Image p, const char* name);
void show_image_mat(mat_cv* mat_ptr, const char* name);

// Data augmentation
Image image_data_augmentation(mat_cv* mat, int w, int h, int pleft, int ptop,
    int swidth, int sheight, int flip, float dhue, float dsat, float dexp,
    int gaussian_noise, int blur, int num_boxes, float* truth);

// blend two images with (alpha and beta)
void blend_images_cv(Image new_img, float alpha, Image old_img, float beta);