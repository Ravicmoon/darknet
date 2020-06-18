#include "data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <iostream>

#include "box.h"
#include "dark_cuda.h"
#include "image.h"
#include "image_opencv.h"
#include "utils.h"

#define NUMCHARS 37

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

list* get_paths(char* filename)
{
  char* path;
  FILE* file = fopen(filename, "r");
  if (!file)
    FileError(filename);
  list* lines = MakeList();
  while ((path = fgetl(file)))
  {
    InsertList(lines, path);
  }
  fclose(file);
  return lines;
}

char** get_random_paths(char** paths, int n, int m)
{
  char** random_paths = (char**)xcalloc(n, sizeof(char*));
  int i;
  pthread_mutex_lock(&mutex);
  // printf("n = %d \n", n);
  for (i = 0; i < n; ++i)
  {
    do
    {
      int index = random_gen() % m;
      random_paths[i] = paths[index];
      // if(i == 0) printf("%s\n", paths[index]);
      // printf("grp: %s\n", paths[index]);
      if (strlen(random_paths[i]) <= 4)
        printf(" Very small path to the image: %s \n", random_paths[i]);
    } while (strlen(random_paths[i]) == 0);
  }
  pthread_mutex_unlock(&mutex);
  return random_paths;
}

matrix load_image_paths(char** paths, int n, int w, int h)
{
  int i;
  matrix X;
  X.rows = n;
  X.vals = (float**)xcalloc(X.rows, sizeof(float*));
  X.cols = 0;

  for (i = 0; i < n; ++i)
  {
    Image im = load_image_color(paths[i], w, h);
    X.vals[i] = im.data;
    X.cols = im.h * im.w * im.c;
  }
  return X;
}

matrix load_image_augment_paths(char** paths, int n, int use_flip, int min,
    int max, int w, int h, float angle, float aspect, float hue,
    float saturation, float exposure, int dontuse_opencv)
{
  int i;
  matrix X;
  X.rows = n;
  X.vals = (float**)xcalloc(X.rows, sizeof(float*));
  X.cols = 0;

  for (i = 0; i < n; ++i)
  {
    int size = w > h ? w : h;
    Image im;
    if (dontuse_opencv)
      im = load_image_stb_resize(paths[i], 0, 0, 3);
    else
      im = load_image_color(paths[i], 0, 0);

    Image crop = random_augment_image(im, angle, aspect, min, max, size);
    int flip = use_flip ? random_gen() % 2 : 0;
    if (flip)
      flip_image(crop);
    random_distort_image(crop, hue, saturation, exposure);

    Image sized = resize_image(crop, w, h);

    // show_image(im, "orig");
    // show_image(sized, "sized");
    // show_image(sized, paths[i]);
    // wait_until_press_key_cv();
    // printf("w = %d, h = %d \n", sized.w, sized.h);

    free_image(im);
    free_image(crop);
    X.vals[i] = sized.data;
    X.cols = sized.h * sized.w * sized.c;
  }
  return X;
}

std::vector<BoxLabel> ReadBoxAnnot(std::string filename)
{
  std::vector<BoxLabel> annot;

  FILE* file = fopen(filename.c_str(), "r");
  if (file == nullptr)
  {
    std::cout << "Cannot open label file: " << filename << std::endl;

    FILE* file_bad = fopen("bad.list", "a");
    fprintf(file_bad, "%s\n", filename.c_str());
    fclose(file_bad);

    return annot;
  }

  int id = 0;
  float x = 0.0f, y = 0.0f, h = 0.0f, w = 0.0f;
  while (fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5)
  {
    BoxLabel box;
    box.id = id;
    box.x = x;
    box.y = y;
    box.h = h;
    box.w = w;
    box.left = x - w / 2;
    box.right = x + w / 2;
    box.top = y - h / 2;
    box.bottom = y + h / 2;

    annot.push_back(box);
  }
  fclose(file);

  return annot;
}

void CorrectBoxAnnot(std::vector<BoxLabel>& annot, float dx, float dy, float sx,
    float sy, int flip)
{
  for (size_t i = 0; i < annot.size(); ++i)
  {
    if (annot[i].x == 0 && annot[i].y == 0)
    {
      annot[i].x = 999999;
      annot[i].y = 999999;
      annot[i].w = 999999;
      annot[i].h = 999999;
      continue;
    }
    if ((annot[i].x + annot[i].w / 2) < 0 ||
        (annot[i].y + annot[i].h / 2) < 0 ||
        (annot[i].x - annot[i].w / 2) > 1 || (annot[i].y - annot[i].h / 2) > 1)
    {
      annot[i].x = 999999;
      annot[i].y = 999999;
      annot[i].w = 999999;
      annot[i].h = 999999;
      continue;
    }
    annot[i].left = annot[i].left * sx - dx;
    annot[i].right = annot[i].right * sx - dx;
    annot[i].top = annot[i].top * sy - dy;
    annot[i].bottom = annot[i].bottom * sy - dy;

    if (flip)
    {
      float swap = annot[i].left;
      annot[i].left = 1. - annot[i].right;
      annot[i].right = 1. - swap;
    }

    annot[i].left = constrain(0, 1, annot[i].left);
    annot[i].right = constrain(0, 1, annot[i].right);
    annot[i].top = constrain(0, 1, annot[i].top);
    annot[i].bottom = constrain(0, 1, annot[i].bottom);

    annot[i].x = (annot[i].left + annot[i].right) / 2;
    annot[i].y = (annot[i].top + annot[i].bottom) / 2;
    annot[i].w = (annot[i].right - annot[i].left);
    annot[i].h = (annot[i].bottom - annot[i].top);

    annot[i].w = constrain(0, 1, annot[i].w);
    annot[i].h = constrain(0, 1, annot[i].h);
  }
}

int fill_truth_detection(const char* path, int num_boxes, float* truth,
    int classes, int flip, float dx, float dy, float sx, float sy, int net_w,
    int net_h)
{
  std::string label_path = ReplaceImage2Label(path);
  std::vector<BoxLabel> annot = ReadBoxAnnot(label_path);
  std::random_shuffle(annot.begin(), annot.end());
  CorrectBoxAnnot(annot, dx, dy, sx, sy, flip);

  int min_w_h = 0;
  float lowest_w = 1.0f / net_w;
  float lowest_h = 1.0f / net_h;

  int id = 0, sub = 0;
  float x = 0.0f, y = 0.0f, w = 0.0f, h = 0.0f;

  for (size_t i = 0; i < std::min(num_boxes, (int)annot.size()); ++i)
  {
    id = annot[i].id;
    x = annot[i].x;
    y = annot[i].y;
    w = annot[i].w;
    h = annot[i].h;

    char buff[256];
    if (id >= classes)
    {
      printf(
          "\n Wrong annotation: class_id = %d. But class_id should be [from 0 "
          "to %d], file: %s \n",
          id, (classes - 1), label_path);
      sprintf(buff,
          "echo %s \"Wrong annotation: class_id = %d. But class_id should "
          "be [from 0 to %d]\" >> bad_label.list",
          label_path, id, (classes - 1));
      system(buff);
      ++sub;
      continue;
    }
    if (w < lowest_w || h < lowest_h)
    {
      // sprintf(buff, "echo %s \"Very small object: w < lowest_w OR h <
      // lowest_h\" >> bad_label.list", labelpath); system(buff);
      ++sub;
      continue;
    }
    if (x == 999999 || y == 999999)
    {
      printf("\n Wrong annotation: x = 0, y = 0, < 0 or > 1, file: %s \n",
          label_path);
      sprintf(buff,
          "echo %s \"Wrong annotation: x = 0 or y = 0\" >> bad_label.list",
          label_path);
      system(buff);
      ++sub;
      continue;
    }
    if (x <= 0 || x > 1 || y <= 0 || y > 1)
    {
      printf(
          "\n Wrong annotation: x = %f, y = %f, file: %s \n", x, y, label_path);
      sprintf(buff,
          "echo %s \"Wrong annotation: x = %f, y = %f\" >> bad_label.list",
          label_path, x, y);
      system(buff);
      ++sub;
      continue;
    }
    if (w > 1)
    {
      printf("\n Wrong annotation: w = %f, file: %s \n", w, label_path);
      sprintf(buff, "echo %s \"Wrong annotation: w = %f\" >> bad_label.list",
          label_path, w);
      system(buff);
      w = 1;
    }
    if (h > 1)
    {
      printf("\n Wrong annotation: h = %f, file: %s \n", h, label_path);
      sprintf(buff, "echo %s \"Wrong annotation: h = %f\" >> bad_label.list",
          label_path, h);
      system(buff);
      h = 1;
    }
    if (x == 0)
      x += lowest_w;
    if (y == 0)
      y += lowest_h;

    truth[(i - sub) * 5 + 0] = x;
    truth[(i - sub) * 5 + 1] = y;
    truth[(i - sub) * 5 + 2] = w;
    truth[(i - sub) * 5 + 3] = h;
    truth[(i - sub) * 5 + 4] = id;

    if (min_w_h == 0)
      min_w_h = w * net_w;
    if (min_w_h > w * net_w)
      min_w_h = w * net_w;
    if (min_w_h > h * net_h)
      min_w_h = h * net_h;
  }

  return min_w_h;
}

void fill_truth_smooth(
    char* path, char** labels, int k, float* truth, float label_smooth_eps)
{
  int i;
  memset(truth, 0, k * sizeof(float));
  int count = 0;
  for (i = 0; i < k; ++i)
  {
    if (strstr(path, labels[i]))
    {
      truth[i] = (1 - label_smooth_eps);
      ++count;
    }
    else
    {
      truth[i] = label_smooth_eps / (k - 1);
    }
  }
  if (count != 1)
  {
    printf("Too many or too few labels: %d, %s\n", count, path);
    count = 0;
    for (i = 0; i < k; ++i)
    {
      if (strstr(path, labels[i]))
      {
        printf("\t label %d: %s  \n", count, labels[i]);
        count++;
      }
    }
  }
}

void fill_hierarchy(float* truth, int k, tree* hierarchy)
{
  int j;
  for (j = 0; j < k; ++j)
  {
    if (truth[j])
    {
      int parent = hierarchy->parent[j];
      while (parent >= 0)
      {
        truth[parent] = 1;
        parent = hierarchy->parent[parent];
      }
    }
  }
  int i;
  int count = 0;
  for (j = 0; j < hierarchy->groups; ++j)
  {
    // printf("%d\n", count);
    int mask = 1;
    for (i = 0; i < hierarchy->group_size[j]; ++i)
    {
      if (truth[count + i])
      {
        mask = 0;
        break;
      }
    }
    if (mask)
    {
      for (i = 0; i < hierarchy->group_size[j]; ++i)
      {
        truth[count + i] = SECRET_NUM;
      }
    }
    count += hierarchy->group_size[j];
  }
}

matrix load_labels_paths(char** paths, int n, char** labels, int k,
    tree* hierarchy, float label_smooth_eps)
{
  matrix y = make_matrix(n, k);
  int i;
  for (i = 0; i < n && labels; ++i)
  {
    fill_truth_smooth(paths[i], labels, k, y.vals[i], label_smooth_eps);
    if (hierarchy)
    {
      fill_hierarchy(y.vals[i], k, hierarchy);
    }
  }
  return y;
}

char** GetLabels(char* filename, int* size)
{
  list* list = get_paths(filename);
  if (size)
    *size = list->size;
  char** labels = (char**)ListToArray(list);
  FreeList(list);
  return labels;
}

void free_data(data d)
{
  if (!d.shallow)
  {
    free_matrix(d.X);
    free_matrix(d.y);
  }
  else
  {
    free(d.X.vals);
    free(d.y.vals);
  }
}

void blend_truth(float* new_truth, int boxes, float* old_truth)
{
  const int t_size = 4 + 1;
  int count_new_truth = 0;
  int t;
  for (t = 0; t < boxes; ++t)
  {
    float x = new_truth[t * (4 + 1)];
    if (!x)
      break;
    count_new_truth++;
  }
  for (t = count_new_truth; t < boxes; ++t)
  {
    float* new_truth_ptr = new_truth + t * t_size;
    float* old_truth_ptr = old_truth + (t - count_new_truth) * t_size;
    float x = old_truth_ptr[0];
    if (!x)
      break;

    new_truth_ptr[0] = old_truth_ptr[0];
    new_truth_ptr[1] = old_truth_ptr[1];
    new_truth_ptr[2] = old_truth_ptr[2];
    new_truth_ptr[3] = old_truth_ptr[3];
    new_truth_ptr[4] = old_truth_ptr[4];
  }
  // printf("\n was %d bboxes, now %d bboxes \n", count_new_truth, t);
}

void blend_truth_mosaic(float* new_truth, int boxes, float* old_truth, int w,
    int h, float cut_x, float cut_y, int i_mixup, int left_shift,
    int right_shift, int top_shift, int bot_shift)
{
  const int t_size = 4 + 1;
  int count_new_truth = 0;
  int t;
  for (t = 0; t < boxes; ++t)
  {
    float x = new_truth[t * (4 + 1)];
    if (!x)
      break;
    count_new_truth++;
  }
  int new_t = count_new_truth;
  for (t = count_new_truth; t < boxes; ++t)
  {
    float* new_truth_ptr = new_truth + new_t * t_size;
    new_truth_ptr[0] = 0;
    float* old_truth_ptr = old_truth + (t - count_new_truth) * t_size;
    float x = old_truth_ptr[0];
    if (!x)
      break;

    float xb = old_truth_ptr[0];
    float yb = old_truth_ptr[1];
    float wb = old_truth_ptr[2];
    float hb = old_truth_ptr[3];

    // shift 4 images
    if (i_mixup == 0)
    {
      xb = xb - (float)(w - cut_x - right_shift) / w;
      yb = yb - (float)(h - cut_y - bot_shift) / h;
    }
    if (i_mixup == 1)
    {
      xb = xb + (float)(cut_x - left_shift) / w;
      yb = yb - (float)(h - cut_y - bot_shift) / h;
    }
    if (i_mixup == 2)
    {
      xb = xb - (float)(w - cut_x - right_shift) / w;
      yb = yb + (float)(cut_y - top_shift) / h;
    }
    if (i_mixup == 3)
    {
      xb = xb + (float)(cut_x - left_shift) / w;
      yb = yb + (float)(cut_y - top_shift) / h;
    }

    int left = (xb - wb / 2) * w;
    int right = (xb + wb / 2) * w;
    int top = (yb - hb / 2) * h;
    int bot = (yb + hb / 2) * h;

    // fix out of bound
    if (left < 0)
    {
      float diff = (float)left / w;
      xb = xb - diff / 2;
      wb = wb + diff;
    }

    if (right > w)
    {
      float diff = (float)(right - w) / w;
      xb = xb - diff / 2;
      wb = wb - diff;
    }

    if (top < 0)
    {
      float diff = (float)top / h;
      yb = yb - diff / 2;
      hb = hb + diff;
    }

    if (bot > h)
    {
      float diff = (float)(bot - h) / h;
      yb = yb - diff / 2;
      hb = hb - diff;
    }

    left = (xb - wb / 2) * w;
    right = (xb + wb / 2) * w;
    top = (yb - hb / 2) * h;
    bot = (yb + hb / 2) * h;

    // leave only within the image
    if (left >= 0 && right <= w && top >= 0 && bot <= h && wb > 0 && wb < 1 &&
        hb > 0 && hb < 1 && xb > 0 && xb < 1 && yb > 0 && yb < 1)
    {
      new_truth_ptr[0] = xb;
      new_truth_ptr[1] = yb;
      new_truth_ptr[2] = wb;
      new_truth_ptr[3] = hb;
      new_truth_ptr[4] = old_truth_ptr[4];
      new_t++;
    }
  }
  // printf("\n was %d bboxes, now %d bboxes \n", count_new_truth, t);
}

data load_data_detection(int n, char** paths, int m, int w, int h, int c,
    int boxes, int classes, int use_flip, int use_gaussian_noise, int use_blur,
    int use_mixup, float jitter, float hue, float saturation, float exposure,
    int mini_batch, int letter_box, int show_imgs)
{
  const int random_index = random_gen();
  c = c ? c : 3;

  if (use_mixup == 2)
  {
    printf("\n cutmix=1 - isn't supported for Detector \n");
    exit(0);
  }
  if (use_mixup == 3 && letter_box)
  {
    printf(
        "\n Combination: letter_box=1 & mosaic=1 - isn't supported, use only 1 "
        "of these parameters \n");
    exit(0);
  }
  if (random_gen() % 2 == 0)
    use_mixup = 0;
  int i;

  int *cut_x = NULL, *cut_y = NULL;
  if (use_mixup == 3)
  {
    cut_x = (int*)calloc(n, sizeof(int));
    cut_y = (int*)calloc(n, sizeof(int));
    const float min_offset = 0.2;  // 20%
    for (i = 0; i < n; ++i)
    {
      cut_x[i] = rand_int(w * min_offset, w * (1 - min_offset));
      cut_y[i] = rand_int(h * min_offset, h * (1 - min_offset));
    }
  }

  data d = {0};
  d.shallow = 0;

  d.X.rows = n;
  d.X.vals = (float**)xcalloc(d.X.rows, sizeof(float*));
  d.X.cols = h * w * c;

  float r1 = 0, r2 = 0, r3 = 0, r4 = 0;
  float dhue = 0, dsat = 0, dexp = 0, flip = 0, blur = 0;
  int gaussian_noise = 0;

  d.y = make_matrix(n, 5 * boxes);
  for (int i_mixup = 0; i_mixup <= use_mixup; i_mixup++)
  {
    char** random_paths = get_random_paths(paths, n, m);

    for (i = 0; i < n; ++i)
    {
      float* truth = (float*)xcalloc(5 * boxes, sizeof(float));
      const char* filename = random_paths[i];

      int flag = (c >= 3);
      mat_cv* src = load_image_mat_cv(filename, flag);
      if (src == NULL)
        continue;

      int oh = get_height_mat(src);
      int ow = get_width_mat(src);

      int dw = (ow * jitter);
      int dh = (oh * jitter);

      r1 = random_float();
      r2 = random_float();
      r3 = random_float();
      r4 = random_float();

      dhue = rand_uniform_strong(-hue, hue);
      dsat = RandScale(saturation);
      dexp = RandScale(exposure);

      flip = use_flip ? random_gen() % 2 : 0;

      if (use_blur)
      {
        int tmp_blur = rand_int(0,
            2);  // 0 - disable, 1 - blur background, 2 - blur the whole image
        if (tmp_blur == 0)
          blur = 0;
        else if (tmp_blur == 1)
          blur = 1;
        else
          blur = use_blur;
      }

      if (use_gaussian_noise && rand_int(0, 1) == 1)
        gaussian_noise = use_gaussian_noise;
      else
        gaussian_noise = 0;

      int pleft = rand_precalc_random(-dw, dw, r1);
      int pright = rand_precalc_random(-dw, dw, r2);
      int ptop = rand_precalc_random(-dh, dh, r3);
      int pbot = rand_precalc_random(-dh, dh, r4);
      // printf("\n pleft = %d, pright = %d, ptop = %d, pbot = %d, ow = %d, oh =
      // %d \n", pleft, pright, ptop, pbot, ow, oh);

      // float scale = rand_precalc_random(.25, 2, r_scale); // unused currently

      if (letter_box)
      {
        float img_ar = (float)ow / (float)oh;
        float net_ar = (float)w / (float)h;
        float result_ar = img_ar / net_ar;
        // printf(" ow = %d, oh = %d, w = %d, h = %d, img_ar = %f, net_ar = %f,
        // result_ar = %f \n", ow, oh, w, h, img_ar, net_ar, result_ar);
        if (result_ar > 1)  // sheight - should be increased
        {
          float oh_tmp = ow / net_ar;
          float delta_h = (oh_tmp - oh) / 2;
          ptop = ptop - delta_h;
          pbot = pbot - delta_h;
          // printf(" result_ar = %f, oh_tmp = %f, delta_h = %d, ptop = %f, pbot
          // = %f \n", result_ar, oh_tmp, delta_h, ptop, pbot);
        }
        else  // swidth - should be increased
        {
          float ow_tmp = oh * net_ar;
          float delta_w = (ow_tmp - ow) / 2;
          pleft = pleft - delta_w;
          pright = pright - delta_w;
          // printf(" result_ar = %f, ow_tmp = %f, delta_w = %d, pleft = %f,
          // pright = %f \n", result_ar, ow_tmp, delta_w, pleft, pright);
        }
      }

      int swidth = ow - pleft - pright;
      int sheight = oh - ptop - pbot;

      float sx = (float)swidth / ow;
      float sy = (float)sheight / oh;

      float dx = ((float)pleft / ow) / sx;
      float dy = ((float)ptop / oh) / sy;

      int min_w_h = fill_truth_detection(filename, boxes, truth, classes, flip,
          dx, dy, 1. / sx, 1. / sy, w, h);

      if ((min_w_h / 8) < blur && blur > 1)
        blur = min_w_h / 8;  // disable blur if one of the objects is too small

      Image ai = image_data_augmentation(src, w, h, pleft, ptop, swidth,
          sheight, flip, dhue, dsat, dexp, gaussian_noise, blur, boxes, truth);

      if (use_mixup == 0)
      {
        d.X.vals[i] = ai.data;
        memcpy(d.y.vals[i], truth, 5 * boxes * sizeof(float));
      }
      else if (use_mixup == 1)
      {
        if (i_mixup == 0)
        {
          d.X.vals[i] = ai.data;
          memcpy(d.y.vals[i], truth, 5 * boxes * sizeof(float));
        }
        else if (i_mixup == 1)
        {
          Image old_img = make_empty_image(w, h, c);
          old_img.data = d.X.vals[i];
          // show_image(ai, "new");
          // show_image(old_img, "old");
          // wait_until_press_key_cv();
          blend_images_cv(ai, 0.5, old_img, 0.5);
          blend_truth(d.y.vals[i], boxes, truth);
          free_image(old_img);
          d.X.vals[i] = ai.data;
        }
      }
      else if (use_mixup == 3)
      {
        if (i_mixup == 0)
        {
          Image tmp_img = make_image(w, h, c);
          d.X.vals[i] = tmp_img.data;
        }

        if (flip)
        {
          int tmp = pleft;
          pleft = pright;
          pright = tmp;
        }

        const int left_shift =
            min_val_cmp(cut_x[i], max_val_cmp(0, (-pleft * w / ow)));
        const int top_shift =
            min_val_cmp(cut_y[i], max_val_cmp(0, (-ptop * h / oh)));

        const int right_shift =
            min_val_cmp((w - cut_x[i]), max_val_cmp(0, (-pright * w / ow)));
        const int bot_shift =
            min_val_cmp(h - cut_y[i], max_val_cmp(0, (-pbot * h / oh)));

        int k, y;
        for (k = 0; k < c; ++k)
        {
          for (y = 0; y < h; ++y)
          {
            int j = y * w + k * w * h;
            if (i_mixup == 0 && y < cut_y[i])
            {
              int j_src = (w - cut_x[i] - right_shift) +
                          (y + h - cut_y[i] - bot_shift) * w + k * w * h;
              memcpy(&d.X.vals[i][j + 0], &ai.data[j_src],
                  cut_x[i] * sizeof(float));
            }
            if (i_mixup == 1 && y < cut_y[i])
            {
              int j_src =
                  left_shift + (y + h - cut_y[i] - bot_shift) * w + k * w * h;
              memcpy(&d.X.vals[i][j + cut_x[i]], &ai.data[j_src],
                  (w - cut_x[i]) * sizeof(float));
            }
            if (i_mixup == 2 && y >= cut_y[i])
            {
              int j_src = (w - cut_x[i] - right_shift) +
                          (top_shift + y - cut_y[i]) * w + k * w * h;
              memcpy(&d.X.vals[i][j + 0], &ai.data[j_src],
                  cut_x[i] * sizeof(float));
            }
            if (i_mixup == 3 && y >= cut_y[i])
            {
              int j_src =
                  left_shift + (top_shift + y - cut_y[i]) * w + k * w * h;
              memcpy(&d.X.vals[i][j + cut_x[i]], &ai.data[j_src],
                  (w - cut_x[i]) * sizeof(float));
            }
          }
        }

        blend_truth_mosaic(d.y.vals[i], boxes, truth, w, h, cut_x[i], cut_y[i],
            i_mixup, left_shift, right_shift, top_shift, bot_shift);

        free_image(ai);
        ai.data = d.X.vals[i];
      }

      if (show_imgs && i_mixup == use_mixup)  // delete i_mixup
      {
        Image tmp_ai = copy_image(ai);
        char buff[1000];
        // sprintf(buff, "aug_%d_%d_%s_%d", random_index, i,
        // basecfg((char*)filename), random_gen());
        sprintf(buff, "aug_%d_%d_%d", random_index, i, random_gen());
        int t;
        for (t = 0; t < boxes; ++t)
        {
          Box b(d.y.vals[i] + t * (4 + 1));
          if (!b.x)
            break;
          int left = (b.x - b.w / 2.) * ai.w;
          int right = (b.x + b.w / 2.) * ai.w;
          int top = (b.y - b.h / 2.) * ai.h;
          int bot = (b.y + b.h / 2.) * ai.h;
          draw_box_width(tmp_ai, left, top, right, bot, 1, 150, 100,
              50);  // 3 channels RGB
        }

        save_image(tmp_ai, buff);
        if (show_imgs == 1)
        {
          // char buff_src[1000];
          // sprintf(buff_src, "src_%d_%d_%s_%d", random_index, i,
          // basecfg((char*)filename), random_gen()); show_image_mat(src,
          // buff_src);
          show_image(tmp_ai, buff);
          wait_until_press_key_cv();
        }
        printf(
            "\nYou use flag -show_imgs, so will be saved aug_...jpg images. "
            "Click on window and press ESC button \n");
        free_image(tmp_ai);
      }

      release_mat(&src);
      free(truth);
    }
    if (random_paths)
      free(random_paths);
  }

  return d;
}

void* load_thread(void* ptr)
{
  // srand(time(0));
  // printf("Loading data: %d\n", random_gen());
  load_args a = *(struct load_args*)ptr;
  if (a.exposure == 0)
    a.exposure = 1;
  if (a.saturation == 0)
    a.saturation = 1;
  if (a.aspect == 0)
    a.aspect = 1;

  if (a.type == CLASSIFICATION_DATA)
  {
    *a.d = load_data_augment(a.paths, a.n, a.m, a.labels, a.classes,
        a.hierarchy, a.flip, a.min, a.max, a.w, a.h, a.angle, a.aspect, a.hue,
        a.saturation, a.exposure, a.mixup, a.blur, a.show_imgs,
        a.label_smooth_eps, a.dontuse_opencv);
  }
  if (a.type == DETECTION_DATA)
  {
    *a.d = load_data_detection(a.n, a.paths, a.m, a.w, a.h, a.c, a.num_boxes,
        a.classes, a.flip, a.gaussian_noise, a.blur, a.mixup, a.jitter, a.hue,
        a.saturation, a.exposure, a.mini_batch, a.letter_box, a.show_imgs);
  }
  if (a.type == IMAGE_DATA)
  {
    *(a.im) = load_image(a.path, 0, 0, a.c);
    *(a.resized) = resize_image(*(a.im), a.w, a.h);
  }
  if (a.type == LETTERBOX_DATA)
  {
    *(a.im) = load_image(a.path, 0, 0, a.c);
    *(a.resized) = letterbox_image(*(a.im), a.w, a.h);
  }

  free(ptr);
  return 0;
}

pthread_t load_data_in_thread(load_args args)
{
  pthread_t thread;
  struct load_args* ptr = (load_args*)xcalloc(1, sizeof(struct load_args));
  *ptr = args;
  if (pthread_create(&thread, 0, load_thread, ptr))
    error("Thread creation failed");
  return thread;
}

static const int thread_wait_ms = 5;
static volatile int flag_exit;
static volatile int* run_load_data = NULL;
static load_args* args_swap = NULL;
static pthread_t* threads = NULL;

pthread_mutex_t mtx_load_data = PTHREAD_MUTEX_INITIALIZER;

void* run_thread_loop(void* ptr)
{
  const int i = *(int*)ptr;

  while (!custom_atomic_load_int(&flag_exit))
  {
    while (!custom_atomic_load_int(&run_load_data[i]))
    {
      if (custom_atomic_load_int(&flag_exit))
      {
        free(ptr);
        return 0;
      }
      this_thread_sleep_for(thread_wait_ms);
    }

    pthread_mutex_lock(&mtx_load_data);
    load_args* args_local = (load_args*)xcalloc(1, sizeof(load_args));
    *args_local = args_swap[i];
    pthread_mutex_unlock(&mtx_load_data);

    load_thread(args_local);

    custom_atomic_store_int(&run_load_data[i], 0);
  }
  free(ptr);
  return 0;
}

void* load_threads(void* ptr)
{
  // srand(time(0));
  int i;
  load_args args = *(load_args*)ptr;
  if (args.threads == 0)
    args.threads = 1;
  data* out = args.d;
  int total = args.n;
  free(ptr);
  data* buffers = (data*)xcalloc(args.threads, sizeof(data));
  if (!threads)
  {
    threads = (pthread_t*)xcalloc(args.threads, sizeof(pthread_t));
    run_load_data = (volatile int*)xcalloc(args.threads, sizeof(int));
    args_swap = (load_args*)xcalloc(args.threads, sizeof(load_args));
    fprintf(stderr, " Create %d permanent cpu-threads \n", args.threads);

    for (i = 0; i < args.threads; ++i)
    {
      int* ptr = (int*)xcalloc(1, sizeof(int));
      *ptr = i;
      if (pthread_create(&threads[i], 0, run_thread_loop, ptr))
        error("Thread creation failed");
    }
  }

  for (i = 0; i < args.threads; ++i)
  {
    args.d = buffers + i;
    args.n = (i + 1) * total / args.threads - i * total / args.threads;

    pthread_mutex_lock(&mtx_load_data);
    args_swap[i] = args;
    pthread_mutex_unlock(&mtx_load_data);

    custom_atomic_store_int(&run_load_data[i], 1);  // run thread
  }
  for (i = 0; i < args.threads; ++i)
  {
    while (custom_atomic_load_int(&run_load_data[i]))
      this_thread_sleep_for(thread_wait_ms);  //   join
  }

  /*
  pthread_t* threads = (pthread_t*)xcalloc(args.threads, sizeof(pthread_t));
  for(i = 0; i < args.threads; ++i){
      args.d = buffers + i;
      args.n = (i+1) * total/args.threads - i * total/args.threads;
      threads[i] = load_data_in_thread(args);
  }
  for(i = 0; i < args.threads; ++i){
      pthread_join(threads[i], 0);
  }
  */

  *out = concat_datas(buffers, args.threads);
  out->shallow = 0;
  for (i = 0; i < args.threads; ++i)
  {
    buffers[i].shallow = 1;
    free_data(buffers[i]);
  }
  free(buffers);
  // free(threads);
  return 0;
}

void free_load_threads(void* ptr)
{
  load_args args = *(load_args*)ptr;
  if (args.threads == 0)
    args.threads = 1;
  int i;
  if (threads)
  {
    custom_atomic_store_int(&flag_exit, 1);
    for (i = 0; i < args.threads; ++i)
    {
      pthread_join(threads[i], 0);
    }
    free((void*)run_load_data);
    free(args_swap);
    free(threads);
    threads = NULL;
    custom_atomic_store_int(&flag_exit, 0);
  }
}

pthread_t load_data(load_args args)
{
  pthread_t thread;
  struct load_args* ptr = (load_args*)xcalloc(1, sizeof(struct load_args));
  *ptr = args;
  if (pthread_create(&thread, 0, load_threads, ptr))
    error("Thread creation failed");
  return thread;
}

data load_data_augment(char** paths, int n, int m, char** labels, int k,
    tree* hierarchy, int use_flip, int min, int max, int w, int h, float angle,
    float aspect, float hue, float saturation, float exposure, int use_mixup,
    int use_blur, int show_imgs, float label_smooth_eps, int dontuse_opencv)
{
  char** paths_stored = paths;
  if (m)
    paths = get_random_paths(paths, n, m);
  data d = {0};
  d.shallow = 0;
  d.X = load_image_augment_paths(paths, n, use_flip, min, max, w, h, angle,
      aspect, hue, saturation, exposure, dontuse_opencv);
  d.y = load_labels_paths(paths, n, labels, k, hierarchy, label_smooth_eps);

  if (use_mixup && rand_int(0, 1))
  {
    char** paths_mix = get_random_paths(paths_stored, n, m);
    data d2 = {0};
    d2.shallow = 0;
    d2.X = load_image_augment_paths(paths_mix, n, use_flip, min, max, w, h,
        angle, aspect, hue, saturation, exposure, dontuse_opencv);
    d2.y =
        load_labels_paths(paths_mix, n, labels, k, hierarchy, label_smooth_eps);
    free(paths_mix);

    data d3 = {0};
    d3.shallow = 0;
    data d4 = {0};
    d4.shallow = 0;
    if (use_mixup >= 3)
    {
      char** paths_mix3 = get_random_paths(paths_stored, n, m);
      d3.X = load_image_augment_paths(paths_mix3, n, use_flip, min, max, w, h,
          angle, aspect, hue, saturation, exposure, dontuse_opencv);
      d3.y = load_labels_paths(
          paths_mix3, n, labels, k, hierarchy, label_smooth_eps);
      free(paths_mix3);

      char** paths_mix4 = get_random_paths(paths_stored, n, m);
      d4.X = load_image_augment_paths(paths_mix4, n, use_flip, min, max, w, h,
          angle, aspect, hue, saturation, exposure, dontuse_opencv);
      d4.y = load_labels_paths(
          paths_mix4, n, labels, k, hierarchy, label_smooth_eps);
      free(paths_mix4);
    }

    // mix
    int i, j;
    for (i = 0; i < d2.X.rows; ++i)
    {
      int mixup = use_mixup;
      if (use_mixup == 4)
        mixup = rand_int(2, 3);  // alternate CutMix and Mosaic

      // MixUp -----------------------------------
      if (mixup == 1)
      {
        // mix images
        for (j = 0; j < d2.X.cols; ++j)
        {
          d.X.vals[i][j] = (d.X.vals[i][j] + d2.X.vals[i][j]) / 2.0f;
        }

        // mix labels
        for (j = 0; j < d2.y.cols; ++j)
        {
          d.y.vals[i][j] = (d.y.vals[i][j] + d2.y.vals[i][j]) / 2.0f;
        }
      }
      // CutMix -----------------------------------
      else if (mixup == 2)
      {
        const float min = 0.3;  // 0.3*0.3 = 9%
        const float max = 0.8;  // 0.8*0.8 = 64%
        const int cut_w = rand_int(w * min, w * max);
        const int cut_h = rand_int(h * min, h * max);
        const int cut_x = rand_int(0, w - cut_w - 1);
        const int cut_y = rand_int(0, h - cut_h - 1);
        const int left = cut_x;
        const int right = cut_x + cut_w;
        const int top = cut_y;
        const int bot = cut_y + cut_h;

        assert(cut_x >= 0 && cut_x <= w);
        assert(cut_y >= 0 && cut_y <= h);
        assert(cut_w >= 0 && cut_w <= w);
        assert(cut_h >= 0 && cut_h <= h);

        assert(right >= 0 && right <= w);
        assert(bot >= 0 && bot <= h);

        assert(top <= bot);
        assert(left <= right);

        const float alpha = (float)(cut_w * cut_h) / (float)(w * h);
        const float beta = 1 - alpha;

        int c, x, y;
        for (c = 0; c < 3; ++c)
        {
          for (y = top; y < bot; ++y)
          {
            for (x = left; x < right; ++x)
            {
              int j = x + y * w + c * w * h;
              d.X.vals[i][j] = d2.X.vals[i][j];
            }
          }
        }

        // printf("\n alpha = %f, beta = %f \n", alpha, beta);
        // mix labels
        for (j = 0; j < d.y.cols; ++j)
        {
          d.y.vals[i][j] = d.y.vals[i][j] * beta + d2.y.vals[i][j] * alpha;
        }
      }
      // Mosaic -----------------------------------
      else if (mixup == 3)
      {
        const float min_offset = 0.2;  // 20%
        const int cut_x = rand_int(w * min_offset, w * (1 - min_offset));
        const int cut_y = rand_int(h * min_offset, h * (1 - min_offset));

        float s1 = (float)(cut_x * cut_y) / (w * h);
        float s2 = (float)((w - cut_x) * cut_y) / (w * h);
        float s3 = (float)(cut_x * (h - cut_y)) / (w * h);
        float s4 = (float)((w - cut_x) * (h - cut_y)) / (w * h);

        int c, x, y;
        for (c = 0; c < 3; ++c)
        {
          for (y = 0; y < h; ++y)
          {
            for (x = 0; x < w; ++x)
            {
              int j = x + y * w + c * w * h;
              if (x < cut_x && y < cut_y)
                d.X.vals[i][j] = d.X.vals[i][j];
              if (x >= cut_x && y < cut_y)
                d.X.vals[i][j] = d2.X.vals[i][j];
              if (x < cut_x && y >= cut_y)
                d.X.vals[i][j] = d3.X.vals[i][j];
              if (x >= cut_x && y >= cut_y)
                d.X.vals[i][j] = d4.X.vals[i][j];
            }
          }
        }

        for (j = 0; j < d.y.cols; ++j)
        {
          const float max_s =
              1;  // max_val_cmp(s1, max_val_cmp(s2, max_val_cmp(s3, s4)));

          d.y.vals[i][j] =
              d.y.vals[i][j] * s1 / max_s + d2.y.vals[i][j] * s2 / max_s +
              d3.y.vals[i][j] * s3 / max_s + d4.y.vals[i][j] * s4 / max_s;
        }
      }
    }

    free_data(d2);

    if (use_mixup >= 3)
    {
      free_data(d3);
      free_data(d4);
    }
  }

  if (use_blur)
  {
    int i;
    for (i = 0; i < d.X.rows; ++i)
    {
      if (random_gen() % 2)
      {
        Image im = make_empty_image(w, h, 3);
        im.data = d.X.vals[i];
        int ksize = use_blur;
        if (use_blur == 1)
          ksize = 17;
        Image blurred = blur_image(im, ksize);
        free_image(im);
        d.X.vals[i] = blurred.data;
        // if (i == 0) {
        //    show_image(im, "Not blurred");
        //    show_image(blurred, "blurred");
        //    wait_until_press_key_cv();
        //}
      }
    }
  }

  if (show_imgs)
  {
    int i, j;
    for (i = 0; i < d.X.rows; ++i)
    {
      Image im = make_empty_image(w, h, 3);
      im.data = d.X.vals[i];
      char buff[1000];
      sprintf(buff, "aug_%d_%s_%d", i, BaseCfg((char*)paths[i]), random_gen());
      save_image(im, buff);

      char buff_string[1000];
      sprintf(buff_string, "\n Classes: ");
      for (j = 0; j < d.y.cols; ++j)
      {
        if (d.y.vals[i][j] > 0)
        {
          char buff_tmp[100];
          sprintf(buff_tmp, " %d (%f), ", j, d.y.vals[i][j]);
          strcat(buff_string, buff_tmp);
        }
      }
      printf("%s \n", buff_string);

      if (show_imgs == 1)
      {
        show_image(im, buff);
        wait_until_press_key_cv();
      }
    }
    printf(
        "\nYou use flag -show_imgs, so will be saved aug_...jpg images. Click "
        "on window and press ESC button \n");
  }

  if (m)
    free(paths);

  return d;
}

matrix concat_matrix(matrix m1, matrix m2)
{
  int i, count = 0;
  matrix m;
  m.cols = m1.cols;
  m.rows = m1.rows + m2.rows;
  m.vals = (float**)xcalloc(m1.rows + m2.rows, sizeof(float*));
  for (i = 0; i < m1.rows; ++i)
  {
    m.vals[count++] = m1.vals[i];
  }
  for (i = 0; i < m2.rows; ++i)
  {
    m.vals[count++] = m2.vals[i];
  }
  return m;
}

data concat_data(data d1, data d2)
{
  data d = {0};
  d.shallow = 1;
  d.X = concat_matrix(d1.X, d2.X);
  d.y = concat_matrix(d1.y, d2.y);
  return d;
}

data concat_datas(data* d, int n)
{
  int i;
  data out = {0};
  for (i = 0; i < n; ++i)
  {
    data newdata = concat_data(d[i], out);
    free_data(out);
    out = newdata;
  }
  return out;
}

void get_random_batch(data d, int n, float* X, float* y)
{
  int j;
  for (j = 0; j < n; ++j)
  {
    int index = random_gen() % d.X.rows;
    memcpy(X + j * d.X.cols, d.X.vals[index], d.X.cols * sizeof(float));
    memcpy(y + j * d.y.cols, d.y.vals[index], d.y.cols * sizeof(float));
  }
}

void get_next_batch(data d, int n, int offset, float* X, float* y)
{
  int j;
  for (j = 0; j < n; ++j)
  {
    int index = offset + j;
    memcpy(X + j * d.X.cols, d.X.vals[index], d.X.cols * sizeof(float));
    memcpy(y + j * d.y.cols, d.y.vals[index], d.y.cols * sizeof(float));
  }
}

data get_data_part(data d, int part, int total)
{
  data p = {0};
  p.shallow = 1;
  p.X.rows = d.X.rows * (part + 1) / total - d.X.rows * part / total;
  p.y.rows = d.y.rows * (part + 1) / total - d.y.rows * part / total;
  p.X.cols = d.X.cols;
  p.y.cols = d.y.cols;
  p.X.vals = d.X.vals + d.X.rows * part / total;
  p.y.vals = d.y.vals + d.y.rows * part / total;
  return p;
}
