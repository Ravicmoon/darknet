#include "data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <fstream>
#include <iostream>

#include "box.h"
#include "dark_cuda.h"
#include "image.h"
#include "image_opencv.h"
#include "utils.h"

#define NUMCHARS 37

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

list* get_paths(char const* filename)
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

std::vector<std::string> GetList(std::string filename)
{
  std::vector<std::string> paths;

  std::ifstream instream(filename);
  while (instream.is_open() && !instream.eof())
  {
    char buffer[256];
    instream.getline(buffer, sizeof(buffer));

    if (std::string(buffer).empty())
      break;

    paths.push_back(buffer);
  }
  instream.close();

  return paths;
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
      int index = RandGen() % m;
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
    int show_imgs)
{
  const int random_index = RandGen();
  c = c ? c : 3;

  if (use_mixup == 2)
  {
    printf("\n cutmix=1 - isn't supported for Detector \n");
    exit(0);
  }
  if (RandGen() % 2 == 0)
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
      cut_x[i] = RandInt(w * min_offset, w * (1 - min_offset));
      cut_y[i] = RandInt(h * min_offset, h * (1 - min_offset));
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

      r1 = RandFloat();
      r2 = RandFloat();
      r3 = RandFloat();
      r4 = RandFloat();

      dhue = RandUniformStrong(-hue, hue);
      dsat = RandScale(saturation);
      dexp = RandScale(exposure);

      flip = use_flip ? RandGen() % 2 : 0;

      if (use_blur)
      {
        int tmp_blur = RandInt(0,
            2);  // 0 - disable, 1 - blur background, 2 - blur the whole image
        if (tmp_blur == 0)
          blur = 0;
        else if (tmp_blur == 1)
          blur = 1;
        else
          blur = use_blur;
      }

      if (use_gaussian_noise && RandInt(0, 1) == 1)
        gaussian_noise = use_gaussian_noise;
      else
        gaussian_noise = 0;

      int pleft = RandPreCalc(-dw, dw, r1);
      int pright = RandPreCalc(-dw, dw, r2);
      int ptop = RandPreCalc(-dh, dh, r3);
      int pbot = RandPreCalc(-dh, dh, r4);
      // printf("\n pleft = %d, pright = %d, ptop = %d, pbot = %d, ow = %d, oh =
      // %d \n", pleft, pright, ptop, pbot, ow, oh);

      // float scale = rand_precalc_random(.25, 2, r_scale); // unused currently

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
        // basecfg((char*)filename), RandGen());
        sprintf(buff, "aug_%d_%d_%d", random_index, i, RandGen());
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
  load_args* args = (load_args*)ptr;
  if (args->exposure == 0)
    args->exposure = 1;
  if (args->saturation == 0)
    args->saturation = 1;
  if (args->aspect == 0)
    args->aspect = 1;

  if (args->type == DETECTION_DATA)
  {
    *args->d = load_data_detection(args->n, args->paths, args->m, args->w,
        args->h, args->c, args->num_boxes, args->classes, args->flip,
        args->gaussian_noise, args->blur, args->mixup, args->jitter, args->hue,
        args->saturation, args->exposure, args->show_imgs);
  }
  if (args->type == IMAGE_DATA)
  {
    *(args->im) = load_image(args->path, 0, 0, args->c);
    *(args->resized) = resize_image(*(args->im), args->w, args->h);
  }

  free(ptr);
  return nullptr;
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
  load_args args = *(load_args*)ptr;
  args.threads = std::max(1, args.threads);
  data* out = args.d;
  int total = args.n;
  free(ptr);
  data* buffers = (data*)xcalloc(args.threads, sizeof(data));
  if (!threads)
  {
    threads = (pthread_t*)xcalloc(args.threads, sizeof(pthread_t));
    run_load_data = (volatile int*)xcalloc(args.threads, sizeof(int));
    args_swap = (load_args*)xcalloc(args.threads, sizeof(load_args));
    fprintf(stderr, " Create %d permanent cpu-threads\n", args.threads);

    for (int i = 0; i < args.threads; ++i)
    {
      int* ptr = (int*)xcalloc(1, sizeof(int));
      *ptr = i;
      if (pthread_create(&threads[i], 0, run_thread_loop, ptr))
        error("Thread creation failed");
    }
  }

  for (int i = 0; i < args.threads; ++i)
  {
    args.d = buffers + i;
    args.n = (i + 1) * total / args.threads - i * total / args.threads;

    pthread_mutex_lock(&mtx_load_data);
    args_swap[i] = args;
    pthread_mutex_unlock(&mtx_load_data);

    custom_atomic_store_int(&run_load_data[i], 1);  // run thread
  }

  for (int i = 0; i < args.threads; ++i)
  {
    while (custom_atomic_load_int(&run_load_data[i]))
    {
      this_thread_sleep_for(thread_wait_ms);  //   join
    }
  }

  *out = concat_datas(buffers, args.threads);
  out->shallow = 0;
  for (int i = 0; i < args.threads; ++i)
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

data GetPartialData(data d, int idx, int num_split)
{
  data p = {0};
  p.shallow = 1;
  p.X.rows = d.X.rows / num_split;
  p.y.rows = d.y.rows / num_split;
  p.X.cols = d.X.cols;
  p.y.cols = d.y.cols;
  p.X.vals = d.X.vals + d.X.rows * idx / num_split;
  p.y.vals = d.y.vals + d.y.rows * idx / num_split;
  return p;
}
