#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "image.h"

#include <stdio.h>

#include "blas.h"
#include "dark_cuda.h"
#include "image_opencv.h"
#include "utils.h"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif

static float get_pixel(Image m, int x, int y, int c)
{
  assert(x < m.w && y < m.h && c < m.c);
  return m.data[c * m.h * m.w + y * m.w + x];
}

static float get_pixel_extend(Image m, int x, int y, int c)
{
  if (x < 0 || x >= m.w || y < 0 || y >= m.h)
    return 0;
  /*
  if(x < 0) x = 0;
  if(x >= m.w) x = m.w-1;
  if(y < 0) y = 0;
  if(y >= m.h) y = m.h-1;
  */
  if (c < 0 || c >= m.c)
    return 0;
  return get_pixel(m, x, y, c);
}

static void set_pixel(Image m, int x, int y, int c, float val)
{
  if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c)
    return;
  assert(x < m.w && y < m.h && c < m.c);
  m.data[c * m.h * m.w + y * m.w + x] = val;
}

static void add_pixel(Image m, int x, int y, int c, float val)
{
  assert(x < m.w && y < m.h && c < m.c);
  m.data[c * m.h * m.w + y * m.w + x] += val;
}

void draw_box(
    Image a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
  if (x1 < 0)
    x1 = 0;
  if (x1 >= a.w)
    x1 = a.w - 1;
  if (x2 < 0)
    x2 = 0;
  if (x2 >= a.w)
    x2 = a.w - 1;

  if (y1 < 0)
    y1 = 0;
  if (y1 >= a.h)
    y1 = a.h - 1;
  if (y2 < 0)
    y2 = 0;
  if (y2 >= a.h)
    y2 = a.h - 1;

  for (int i = x1; i <= x2; ++i)
  {
    a.data[i + y1 * a.w + 0 * a.w * a.h] = r;
    a.data[i + y2 * a.w + 0 * a.w * a.h] = r;

    a.data[i + y1 * a.w + 1 * a.w * a.h] = g;
    a.data[i + y2 * a.w + 1 * a.w * a.h] = g;

    a.data[i + y1 * a.w + 2 * a.w * a.h] = b;
    a.data[i + y2 * a.w + 2 * a.w * a.h] = b;
  }
  for (int i = y1; i <= y2; ++i)
  {
    a.data[x1 + i * a.w + 0 * a.w * a.h] = r;
    a.data[x2 + i * a.w + 0 * a.w * a.h] = r;

    a.data[x1 + i * a.w + 1 * a.w * a.h] = g;
    a.data[x2 + i * a.w + 1 * a.w * a.h] = g;

    a.data[x1 + i * a.w + 2 * a.w * a.h] = b;
    a.data[x2 + i * a.w + 2 * a.w * a.h] = b;
  }
}

void draw_box_width(
    Image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
{
  for (int i = 0; i < w; ++i)
  {
    draw_box(a, x1 + i, y1 + i, x2 - i, y2 - i, r, g, b);
  }
}

void flip_image(Image a)
{
  for (int k = 0; k < a.c; ++k)
  {
    for (int i = 0; i < a.h; ++i)
    {
      for (int j = 0; j < a.w / 2; ++j)
      {
        int idx = j + a.w * (i + a.h * (k));
        int flip = (a.w - j - 1) + a.w * (i + a.h * (k));
        float swap = a.data[flip];
        a.data[flip] = a.data[idx];
        a.data[idx] = swap;
      }
    }
  }
}

void embed_image(Image src, Image dst, int dx, int dy)
{
  for (int k = 0; k < src.c; ++k)
  {
    for (int y = 0; y < src.h; ++y)
    {
      for (int x = 0; x < src.w; ++x)
      {
        float val = get_pixel(src, x, y, k);
        set_pixel(dst, dx + x, dy + y, k, val);
      }
    }
  }
}

void constrain_image(Image im)
{
  for (int i = 0; i < im.w * im.h * im.c; ++i)
  {
    if (im.data[i] < 0)
      im.data[i] = 0;
    if (im.data[i] > 1)
      im.data[i] = 1;
  }
}

Image copy_image(Image p)
{
  Image copy = p;
  copy.data = (float*)xcalloc(p.h * p.w * p.c, sizeof(float));
  memcpy(copy.data, p.data, p.h * p.w * p.c * sizeof(float));
  return copy;
}

void show_image(Image p, const char* name) { show_image_cv(p, name); }

void save_image_options(Image im, const char* name, IMTYPE f, int quality)
{
  char buff[256];
  // sprintf(buff, "%s (%d)", name, windows);
  if (f == PNG)
    sprintf(buff, "%s.png", name);
  else if (f == BMP)
    sprintf(buff, "%s.bmp", name);
  else if (f == TGA)
    sprintf(buff, "%s.tga", name);
  else if (f == JPG)
    sprintf(buff, "%s.jpg", name);
  else
    sprintf(buff, "%s.png", name);
  unsigned char* data =
      (unsigned char*)xcalloc(im.w * im.h * im.c, sizeof(unsigned char));
  int i, k;
  for (k = 0; k < im.c; ++k)
  {
    for (i = 0; i < im.w * im.h; ++i)
    {
      data[i * im.c + k] = (unsigned char)(255 * im.data[i + k * im.w * im.h]);
    }
  }
  int success = 0;
  if (f == PNG)
    success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w * im.c);
  else if (f == BMP)
    success = stbi_write_bmp(buff, im.w, im.h, im.c, data);
  else if (f == TGA)
    success = stbi_write_tga(buff, im.w, im.h, im.c, data);
  else if (f == JPG)
    success = stbi_write_jpg(buff, im.w, im.h, im.c, data, quality);
  free(data);
  if (!success)
    printf("Failed to write image %s\n", buff);
}

void save_image(Image im, const char* name)
{
  save_image_options(im, name, JPG, 80);
}

Image make_empty_image(int w, int h, int c)
{
  Image out;
  out.data = 0;
  out.h = h;
  out.w = w;
  out.c = c;
  return out;
}

Image make_image(int w, int h, int c)
{
  Image out = make_empty_image(w, h, c);
  out.data = (float*)xcalloc(h * w * c, sizeof(float));
  return out;
}

Image rotate_crop_image(Image im, float rad, float s, int w, int h, float dx,
    float dy, float aspect)
{
  int x, y, c;
  float cx = im.w / 2.;
  float cy = im.h / 2.;
  Image rot = make_image(w, h, im.c);
  for (c = 0; c < im.c; ++c)
  {
    for (y = 0; y < h; ++y)
    {
      for (x = 0; x < w; ++x)
      {
        float rx = cos(rad) * ((x - w / 2.) / s * aspect + dx / s * aspect) -
                   sin(rad) * ((y - h / 2.) / s + dy / s) + cx;
        float ry = sin(rad) * ((x - w / 2.) / s * aspect + dx / s * aspect) +
                   cos(rad) * ((y - h / 2.) / s + dy / s) + cy;
        float val = bilinear_interpolate(im, rx, ry, c);
        set_pixel(rot, x, y, c, val);
      }
    }
  }
  return rot;
}

void scale_image(Image m, float s)
{
  int i;
  for (i = 0; i < m.h * m.w * m.c; ++i) m.data[i] *= s;
}

Image crop_image(Image im, int dx, int dy, int w, int h)
{
  Image cropped = make_image(w, h, im.c);
  int i, j, k;
  for (k = 0; k < im.c; ++k)
  {
    for (j = 0; j < h; ++j)
    {
      for (i = 0; i < w; ++i)
      {
        int r = j + dy;
        int c = i + dx;
        float val = 0;
        r = constrain_int(r, 0, im.h - 1);
        c = constrain_int(c, 0, im.w - 1);
        if (r >= 0 && r < im.h && c >= 0 && c < im.w)
        {
          val = get_pixel(im, c, r, k);
        }
        set_pixel(cropped, i, j, k, val);
      }
    }
  }
  return cropped;
}

void fill_image(Image m, float s)
{
  int i;
  for (i = 0; i < m.h * m.w * m.c; ++i) m.data[i] = s;
}

Image random_crop_image(Image im, int w, int h)
{
  int dx = RandInt(0, im.w - w);
  int dy = RandInt(0, im.h - h);
  Image crop = crop_image(im, dx, dy, w, h);
  return crop;
}

Image random_augment_image(
    Image im, float angle, float aspect, int low, int high, int size)
{
  aspect = RandScale(aspect);
  int r = RandInt(low, high);
  int min = (im.h < im.w * aspect) ? im.h : im.w * aspect;
  float scale = (float)r / min;

  float rad = RandUniform(-angle, angle) * 2.0 * M_PI / 360.;

  float dx = (im.w * scale / aspect - size) / 2.;
  float dy = (im.h * scale - size) / 2.;
  if (dx < 0)
    dx = 0;
  if (dy < 0)
    dy = 0;
  dx = RandUniform(-dx, dx);
  dy = RandUniform(-dy, dy);

  Image crop = rotate_crop_image(im, rad, scale, size, size, dx, dy, aspect);

  return crop;
}

float three_way_max(float a, float b, float c)
{
  return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c);
}

float three_way_min(float a, float b, float c)
{
  return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c);
}

// http://www.cs.rit.edu/~ncs/color/t_convert.html
void rgb_to_hsv(Image im)
{
  assert(im.c == 3);
  int i, j;
  float r, g, b;
  float h, s, v;
  for (j = 0; j < im.h; ++j)
  {
    for (i = 0; i < im.w; ++i)
    {
      r = get_pixel(im, i, j, 0);
      g = get_pixel(im, i, j, 1);
      b = get_pixel(im, i, j, 2);
      float max = three_way_max(r, g, b);
      float min = three_way_min(r, g, b);
      float delta = max - min;
      v = max;
      if (max == 0)
      {
        s = 0;
        h = 0;
      }
      else
      {
        s = delta / max;
        if (r == max)
        {
          h = (g - b) / delta;
        }
        else if (g == max)
        {
          h = 2 + (b - r) / delta;
        }
        else
        {
          h = 4 + (r - g) / delta;
        }
        if (h < 0)
          h += 6;
        h = h / 6.;
      }
      set_pixel(im, i, j, 0, h);
      set_pixel(im, i, j, 1, s);
      set_pixel(im, i, j, 2, v);
    }
  }
}

void hsv_to_rgb(Image im)
{
  assert(im.c == 3);
  int i, j;
  float r, g, b;
  float h, s, v;
  float f, p, q, t;
  for (j = 0; j < im.h; ++j)
  {
    for (i = 0; i < im.w; ++i)
    {
      h = 6 * get_pixel(im, i, j, 0);
      s = get_pixel(im, i, j, 1);
      v = get_pixel(im, i, j, 2);
      if (s == 0)
      {
        r = g = b = v;
      }
      else
      {
        int index = floor(h);
        f = h - index;
        p = v * (1 - s);
        q = v * (1 - s * f);
        t = v * (1 - s * (1 - f));
        if (index == 0)
        {
          r = v;
          g = t;
          b = p;
        }
        else if (index == 1)
        {
          r = q;
          g = v;
          b = p;
        }
        else if (index == 2)
        {
          r = p;
          g = v;
          b = t;
        }
        else if (index == 3)
        {
          r = p;
          g = q;
          b = v;
        }
        else if (index == 4)
        {
          r = t;
          g = p;
          b = v;
        }
        else
        {
          r = v;
          g = p;
          b = q;
        }
      }
      set_pixel(im, i, j, 0, r);
      set_pixel(im, i, j, 1, g);
      set_pixel(im, i, j, 2, b);
    }
  }
}

void scale_image_channel(Image im, int c, float v)
{
  int i, j;
  for (j = 0; j < im.h; ++j)
  {
    for (i = 0; i < im.w; ++i)
    {
      float pix = get_pixel(im, i, j, c);
      pix = pix * v;
      set_pixel(im, i, j, c, pix);
    }
  }
}

void distort_image(Image im, float hue, float sat, float val)
{
  if (im.c >= 3)
  {
    rgb_to_hsv(im);
    scale_image_channel(im, 1, sat);
    scale_image_channel(im, 2, val);
    int i;
    for (i = 0; i < im.w * im.h; ++i)
    {
      im.data[i] = im.data[i] + hue;
      if (im.data[i] > 1)
        im.data[i] -= 1;
      if (im.data[i] < 0)
        im.data[i] += 1;
    }
    hsv_to_rgb(im);
  }
  else
  {
    scale_image_channel(im, 0, val);
  }
  constrain_image(im);
}

void random_distort_image(Image im, float hue, float saturation, float exposure)
{
  float dhue = RandUniformStrong(-hue, hue);
  float dsat = RandScale(saturation);
  float dexp = RandScale(exposure);
  distort_image(im, dhue, dsat, dexp);
}

float bilinear_interpolate(Image im, float x, float y, int c)
{
  int ix = (int)floorf(x);
  int iy = (int)floorf(y);

  float dx = x - ix;
  float dy = y - iy;

  float val = (1 - dy) * (1 - dx) * get_pixel_extend(im, ix, iy, c) +
              dy * (1 - dx) * get_pixel_extend(im, ix, iy + 1, c) +
              (1 - dy) * dx * get_pixel_extend(im, ix + 1, iy, c) +
              dy * dx * get_pixel_extend(im, ix + 1, iy + 1, c);
  return val;
}

void make_image_red(Image im)
{
  int r, c, k;
  for (r = 0; r < im.h; ++r)
  {
    for (c = 0; c < im.w; ++c)
    {
      float val = 0;
      for (k = 0; k < im.c; ++k)
      {
        val += get_pixel(im, c, r, k);
        set_pixel(im, c, r, k, 0);
      }
      for (k = 0; k < im.c; ++k)
      {
        // set_pixel(im, c, r, k, val);
      }
      set_pixel(im, c, r, 0, val);
    }
  }
}

Image make_attention_image(int img_size, float* original_delta_cpu,
    float* original_input_cpu, int w, int h, int c)
{
  Image attention_img;
  attention_img.w = w;
  attention_img.h = h;
  attention_img.c = c;
  attention_img.data = original_delta_cpu;
  make_image_red(attention_img);

  int k;
  float min_val = 999999, mean_val = 0, max_val = -999999;
  for (k = 0; k < img_size; ++k)
  {
    if (original_delta_cpu[k] < min_val)
      min_val = original_delta_cpu[k];
    if (original_delta_cpu[k] > max_val)
      max_val = original_delta_cpu[k];
    mean_val += original_delta_cpu[k];
  }
  mean_val = mean_val / img_size;
  float range = max_val - min_val;

  for (k = 0; k < img_size; ++k)
  {
    float val = original_delta_cpu[k];
    val = fabs(mean_val - val) / range;
    original_delta_cpu[k] = val * 4;
  }

  Image resized = resize_image(attention_img, w / 4, h / 4);
  attention_img = resize_image(resized, w, h);
  free_image(resized);
  for (k = 0; k < img_size; ++k) attention_img.data[k] += original_input_cpu[k];

  // normalize_image(attention_img);
  // show_image(attention_img, "delta");
  return attention_img;
}

Image resize_image(Image im, int w, int h)
{
  if (im.w == w && im.h == h)
    return copy_image(im);

  Image resized = make_image(w, h, im.c);
  Image part = make_image(w, im.h, im.c);
  int r, c, k;
  float w_scale = (float)(im.w - 1) / (w - 1);
  float h_scale = (float)(im.h - 1) / (h - 1);
  for (k = 0; k < im.c; ++k)
  {
    for (r = 0; r < im.h; ++r)
    {
      for (c = 0; c < w; ++c)
      {
        float val = 0;
        if (c == w - 1 || im.w == 1)
        {
          val = get_pixel(im, im.w - 1, r, k);
        }
        else
        {
          float sx = c * w_scale;
          int ix = (int)sx;
          float dx = sx - ix;
          val = (1 - dx) * get_pixel(im, ix, r, k) +
                dx * get_pixel(im, ix + 1, r, k);
        }
        set_pixel(part, c, r, k, val);
      }
    }
  }
  for (k = 0; k < im.c; ++k)
  {
    for (r = 0; r < h; ++r)
    {
      float sy = r * h_scale;
      int iy = (int)sy;
      float dy = sy - iy;
      for (c = 0; c < w; ++c)
      {
        float val = (1 - dy) * get_pixel(part, c, iy, k);
        set_pixel(resized, c, r, k, val);
      }
      if (r == h - 1 || im.h == 1)
        continue;
      for (c = 0; c < w; ++c)
      {
        float val = dy * get_pixel(part, c, iy + 1, k);
        add_pixel(resized, c, r, k, val);
      }
    }
  }

  free_image(part);
  return resized;
}

Image load_image_stb(char const* filename, int channels)
{
  int w, h, c;
  unsigned char* data = stbi_load(filename, &w, &h, &c, channels);
  if (!data)
  {
    char shrinked_filename[1024];
    if (strlen(filename) >= 1024)
      sprintf(shrinked_filename, "name is too long");
    else
      sprintf(shrinked_filename, "%s", filename);
    printf("Cannot load image \"%s\"\nSTB Reason: %s\n", shrinked_filename,
        stbi_failure_reason());
    FILE* fw = fopen("bad.list", "a");
    fwrite(shrinked_filename, sizeof(char), strlen(shrinked_filename), fw);
    char* new_line = "\n";
    fwrite(new_line, sizeof(char), strlen(new_line), fw);
    fclose(fw);
    return make_image(10, 10, 3);
  }
  if (channels)
    c = channels;
  int i, j, k;
  Image im = make_image(w, h, c);
  for (k = 0; k < c; ++k)
  {
    for (j = 0; j < h; ++j)
    {
      for (i = 0; i < w; ++i)
      {
        int dst_index = i + w * j + w * h * k;
        int src_index = k + c * i + c * w * j;
        im.data[dst_index] = (float)data[src_index] / 255.;
      }
    }
  }
  free(data);
  return im;
}

Image load_image_stb_resize(char* filename, int w, int h, int c)
{
  Image out = load_image_stb(filename, c);  // without OpenCV

  if ((h && w) && (h != out.h || w != out.w))
  {
    Image resized = resize_image(out, w, h);
    free_image(out);
    out = resized;
  }
  return out;
}

Image load_image(char const* filename, int w, int h, int c)
{
  Image out = load_image_cv(filename, c);

  if ((h && w) && (h != out.h || w != out.w))
  {
    Image resized = resize_image(out, w, h);
    free_image(out);
    out = resized;
  }
  return out;
}

Image load_image_color(char* filename, int w, int h)
{
  return load_image(filename, w, h, 3);
}

void free_image(Image m)
{
  if (m.data)
  {
    free(m.data);
  }
}