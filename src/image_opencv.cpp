#include "image_opencv.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "utils.h"

using std::cerr;
using std::endl;

#ifdef DEBUG
#define OCV_D "d"
#else
#define OCV_D
#endif  // DEBUG

// OpenCV libraries
#ifndef CV_VERSION_EPOCH
#define OPENCV_VERSION        \
  CVAUX_STR(CV_VERSION_MAJOR) \
  "" CVAUX_STR(CV_VERSION_MINOR) "" CVAUX_STR(CV_VERSION_REVISION) OCV_D
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")
#endif  // USE_CMAKE_LIBS
#else   // CV_VERSION_EPOCH
#define OPENCV_VERSION        \
  CVAUX_STR(CV_VERSION_EPOCH) \
  "" CVAUX_STR(CV_VERSION_MAJOR) "" CVAUX_STR(CV_VERSION_MINOR) OCV_D
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
#endif  // USE_CMAKE_LIBS
#endif  // CV_VERSION_EPOCH

#ifndef CV_RGB
#define CV_RGB(r, g, b) cvScalar((b), (g), (r), 0)
#endif

#ifndef CV_FILLED
#define CV_FILLED cv::FILLED
#endif

#ifndef CV_AA
#define CV_AA cv::LINE_AA
#endif

// ====================================================================
// cv::Mat
// ====================================================================
Image mat_to_image(cv::Mat mat);
cv::Mat image_to_mat(Image img);

mat_cv* load_image_mat_cv(const char* filename, int flag)
{
  cv::Mat* mat_ptr = NULL;
  try
  {
    cv::Mat mat = cv::imread(filename, flag);
    if (mat.empty())
    {
      std::string shrinked_filename = filename;
      if (shrinked_filename.length() > 1024)
      {
        shrinked_filename.resize(1024);
        shrinked_filename =
            std::string("name is too long: ") + shrinked_filename;
      }
      cerr << "Cannot load image " << shrinked_filename << std::endl;
      std::ofstream bad_list("bad.list", std::ios::out | std::ios::app);
      bad_list << shrinked_filename << std::endl;
      return NULL;
    }
    cv::Mat dst;
    if (mat.channels() == 3)
      cv::cvtColor(mat, dst, cv::COLOR_RGB2BGR);
    else if (mat.channels() == 4)
      cv::cvtColor(mat, dst, cv::COLOR_RGBA2BGRA);
    else
      dst = mat;

    mat_ptr = new cv::Mat(dst);

    return (mat_cv*)mat_ptr;
  }
  catch (...)
  {
    cerr << "OpenCV exception: load_image_mat_cv \n";
  }
  if (mat_ptr)
    delete mat_ptr;
  return NULL;
}
// ----------------------------------------

cv::Mat load_image_mat(char const* filename, int channels)
{
  int flag = cv::IMREAD_UNCHANGED;
  if (channels == 0)
    flag = cv::IMREAD_COLOR;
  else if (channels == 1)
    flag = cv::IMREAD_GRAYSCALE;
  else if (channels == 3)
    flag = cv::IMREAD_COLOR;
  else
  {
    fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
  }
  // flag |= IMREAD_IGNORE_ORIENTATION;    // un-comment it if you want

  cv::Mat* mat_ptr = (cv::Mat*)load_image_mat_cv(filename, flag);

  if (mat_ptr == NULL)
  {
    return cv::Mat();
  }
  cv::Mat mat = *mat_ptr;
  delete mat_ptr;

  return mat;
}
// ----------------------------------------

Image load_image_cv(char const* filename, int channels)
{
  cv::Mat mat = load_image_mat(filename, channels);

  if (mat.empty())
  {
    return make_image(10, 10, channels);
  }
  return mat_to_image(mat);
}
// ----------------------------------------

Image load_image_resize(char* filename, int w, int h, int c, Image* im)
{
  Image out;
  try
  {
    cv::Mat loaded_image = load_image_mat(filename, c);

    *im = mat_to_image(loaded_image);

    cv::Mat resized(h, w, CV_8UC3);
    cv::resize(loaded_image, resized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
    out = mat_to_image(resized);
  }
  catch (...)
  {
    cerr << " OpenCV exception: load_image_resize() can't load image %s "
         << filename << " \n";
    out = make_image(w, h, c);
    *im = make_image(w, h, c);
  }
  return out;
}
// ----------------------------------------

int get_width_mat(mat_cv* mat)
{
  if (mat == NULL)
  {
    cerr << " Pointer is NULL in get_width_mat() \n";
    return 0;
  }
  return ((cv::Mat*)mat)->cols;
}
// ----------------------------------------

int get_height_mat(mat_cv* mat)
{
  if (mat == NULL)
  {
    cerr << " Pointer is NULL in get_height_mat() \n";
    return 0;
  }
  return ((cv::Mat*)mat)->rows;
}
// ----------------------------------------

void release_mat(mat_cv** mat)
{
  try
  {
    cv::Mat** mat_ptr = (cv::Mat**)mat;
    if (*mat_ptr)
      delete *mat_ptr;
    *mat_ptr = NULL;
  }
  catch (...)
  {
    cerr << "OpenCV exception: release_mat \n";
  }
}

cv::Mat image_to_mat(Image img)
{
  int channels = img.c;
  int width = img.w;
  int height = img.h;
  cv::Mat mat = cv::Mat(height, width, CV_8UC(channels));
  int step = mat.step;

  for (int y = 0; y < img.h; ++y)
  {
    for (int x = 0; x < img.w; ++x)
    {
      for (int c = 0; c < img.c; ++c)
      {
        float val = img.data[c * img.h * img.w + y * img.w + x];
        mat.data[y * step + x * img.c + c] = (unsigned char)(val * 255);
      }
    }
  }
  return mat;
}
// ----------------------------------------

Image mat_to_image(cv::Mat mat)
{
  int w = mat.cols;
  int h = mat.rows;
  int c = mat.channels();
  Image im = make_image(w, h, c);
  unsigned char* data = (unsigned char*)mat.data;
  int step = mat.step;
  for (int y = 0; y < h; ++y)
  {
    for (int k = 0; k < c; ++k)
    {
      for (int x = 0; x < w; ++x)
      {
        im.data[k * w * h + y * w + x] = data[y * step + x * c + k] / 255.0f;
      }
    }
  }
  return im;
}

// ====================================================================
// Window
// ====================================================================
void create_window_cv(
    char const* window_name, int full_screen, int width, int height)
{
  try
  {
    int window_type = cv::WINDOW_NORMAL;
#ifdef CV_VERSION_EPOCH  // OpenCV 2.x
    if (full_screen)
      window_type = CV_WINDOW_FULLSCREEN;
#else
    if (full_screen)
      window_type = cv::WINDOW_FULLSCREEN;
#endif
    cv::namedWindow(window_name, window_type);
    cv::moveWindow(window_name, 0, 0);
    cv::resizeWindow(window_name, width, height);
  }
  catch (...)
  {
    cerr << "OpenCV exception: create_window_cv \n";
  }
}
// ----------------------------------------

void destroy_all_windows_cv()
{
  try
  {
    cv::destroyAllWindows();
  }
  catch (...)
  {
    cerr << "OpenCV exception: destroy_all_windows_cv \n";
  }
}
// ----------------------------------------

int wait_key_cv(int delay)
{
  try
  {
    return cv::waitKey(delay);
  }
  catch (...)
  {
    cerr << "OpenCV exception: wait_key_cv \n";
  }
  return -1;
}
// ----------------------------------------

int wait_until_press_key_cv() { return wait_key_cv(0); }
// ----------------------------------------

void make_window(char* name, int w, int h, int fullscreen)
{
  try
  {
    cv::namedWindow(name, cv::WINDOW_NORMAL);
    if (fullscreen)
    {
#ifdef CV_VERSION_EPOCH  // OpenCV 2.x
      cv::setWindowProperty(
          name, cv::WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
#else
      cv::setWindowProperty(
          name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
#endif
    }
    else
    {
      cv::resizeWindow(name, w, h);
      if (strcmp(name, "Demo") == 0)
        cv::moveWindow(name, 0, 0);
    }
  }
  catch (...)
  {
    cerr << "OpenCV exception: make_window \n";
  }
}
// ----------------------------------------

void show_image_cv(Image p, const char* name)
{
  try
  {
    Image copy = copy_image(p);
    constrain_image(copy);

    cv::Mat mat = image_to_mat(copy);
    if (mat.channels() == 3)
      cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
    else if (mat.channels() == 4)
      cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGR);
    cv::namedWindow(name, cv::WINDOW_NORMAL);
    cv::imshow(name, mat);
    free_image(copy);
  }
  catch (...)
  {
    cerr << "OpenCV exception: show_image_cv \n";
  }
}
// ----------------------------------------

void show_image_mat(mat_cv* mat_ptr, const char* name)
{
  try
  {
    if (mat_ptr == NULL)
      return;
    cv::Mat& mat = *(cv::Mat*)mat_ptr;
    cv::namedWindow(name, cv::WINDOW_NORMAL);
    cv::imshow(name, mat);
  }
  catch (...)
  {
    cerr << "OpenCV exception: show_image_mat \n";
  }
}

// ====================================================================
// Data augmentation
// ====================================================================

Image image_data_augmentation(mat_cv* mat, int w, int h, int pleft, int ptop,
    int swidth, int sheight, int flip, float dhue, float dsat, float dexp,
    int gaussian_noise, int blur, int num_boxes, float* truth)
{
  Image out;
  try
  {
    cv::Mat img = *(cv::Mat*)mat;

    // crop
    cv::Rect src_rect(pleft, ptop, swidth, sheight);
    cv::Rect img_rect(cv::Point2i(0, 0), img.size());
    cv::Rect new_src_rect = src_rect & img_rect;

    cv::Rect dst_rect(
        cv::Point2i(std::max<int>(0, -pleft), std::max<int>(0, -ptop)),
        new_src_rect.size());
    cv::Mat sized;

    if (src_rect.x == 0 && src_rect.y == 0 && src_rect.size() == img.size())
    {
      cv::resize(img, sized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
    }
    else
    {
      cv::Mat cropped(src_rect.size(), img.type());
      // cropped.setTo(cv::Scalar::all(0));
      cropped.setTo(cv::mean(img));

      img(new_src_rect).copyTo(cropped(dst_rect));

      // resize
      cv::resize(cropped, sized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
    }

    // flip
    if (flip)
    {
      cv::Mat cropped;
      cv::flip(sized, cropped,
          1);  // 0 - x-axis, 1 - y-axis, -1 - both axes (x & y)
      sized = cropped.clone();
    }

    // HSV augmentation
    // cv::COLOR_BGR2HSV, cv::COLOR_RGB2HSV, cv::COLOR_HSV2BGR,
    // cv::COLOR_HSV2RGB
    if (dsat != 1 || dexp != 1 || dhue != 0)
    {
      if (img.channels() >= 3)
      {
        cv::Mat hsv_src;
        cvtColor(sized, hsv_src, cv::COLOR_RGB2HSV);  // RGB to HSV

        std::vector<cv::Mat> hsv;
        cv::split(hsv_src, hsv);

        hsv[1] *= dsat;
        hsv[2] *= dexp;
        hsv[0] += 179 * dhue;

        cv::merge(hsv, hsv_src);

        cvtColor(hsv_src, sized,
            cv::COLOR_HSV2RGB);  // HSV to RGB (the same as previous)
      }
      else
      {
        sized *= dexp;
      }
    }

    // std::stringstream window_name;
    // window_name << "augmentation - " << ipl;
    // cv::imshow(window_name.str(), sized);
    // cv::waitKey(0);

    if (blur)
    {
      cv::Mat dst(sized.size(), sized.type());
      if (blur == 1)
      {
        cv::GaussianBlur(sized, dst, cv::Size(17, 17), 0);
        // cv::bilateralFilter(sized, dst, 17, 75, 75);
      }
      else
      {
        int ksize = (blur / 2) * 2 + 1;
        cv::Size kernel_size = cv::Size(ksize, ksize);
        cv::GaussianBlur(sized, dst, kernel_size, 0);
        // cv::medianBlur(sized, dst, ksize);
        // cv::bilateralFilter(sized, dst, ksize, 75, 75);

        // sharpen
        // cv::Mat img_tmp;
        // cv::GaussianBlur(dst, img_tmp, cv::Size(), 3);
        // cv::addWeighted(dst, 1.5, img_tmp, -0.5, 0, img_tmp);
        // dst = img_tmp;
      }
      // std::cout << " blur num_boxes = " << num_boxes << std::endl;

      if (blur == 1)
      {
        cv::Rect img_rect(0, 0, sized.cols, sized.rows);
        int t;
        for (t = 0; t < num_boxes; ++t)
        {
          Box b(truth + t * (4 + 1));
          if (!b.x)
            break;
          int left = (b.x - b.w / 2.) * sized.cols;
          int width = b.w * sized.cols;
          int top = (b.y - b.h / 2.) * sized.rows;
          int height = b.h * sized.rows;
          cv::Rect roi(left, top, width, height);
          roi = roi & img_rect;

          sized(roi).copyTo(dst(roi));
        }
      }
      dst.copyTo(sized);
    }

    if (gaussian_noise)
    {
      cv::Mat noise = cv::Mat(sized.size(), sized.type());
      gaussian_noise = std::min(gaussian_noise, 127);
      gaussian_noise = std::max(gaussian_noise, 0);
      cv::randn(noise, 0, gaussian_noise);  // mean and variance
      cv::Mat sized_norm = sized + noise;
      // cv::normalize(sized_norm, sized_norm, 0.0, 255.0, cv::NORM_MINMAX,
      // sized.type()); cv::imshow("source", sized); cv::imshow("gaussian
      // noise", sized_norm); cv::waitKey(0);
      sized = sized_norm;
    }

    // char txt[100];
    // sprintf(txt, "blur = %d", blur);
    // cv::putText(sized, txt, cv::Point(100, 100),
    // cv::FONT_HERSHEY_COMPLEX_SMALL, 1.7, CV_RGB(255, 0, 0), 1, CV_AA);

    // Mat -> image
    out = mat_to_image(sized);
  }
  catch (...)
  {
    cerr << "OpenCV can't augment image: " << w << " x " << h << " \n";
    out = mat_to_image(*(cv::Mat*)mat);
  }
  return out;
}

// blend two images with (alpha and beta)
void blend_images_cv(Image new_img, float alpha, Image old_img, float beta)
{
  cv::Mat new_mat(cv::Size(new_img.w, new_img.h), CV_32FC(new_img.c),
      new_img.data);  // , size_t step = AUTO_STEP)
  cv::Mat old_mat(
      cv::Size(old_img.w, old_img.h), CV_32FC(old_img.c), old_img.data);
  cv::addWeighted(new_mat, alpha, old_mat, beta, 0.0, new_mat);
}

// bilateralFilter bluring
Image blur_image(Image src_img, int ksize)
{
  cv::Mat src = image_to_mat(src_img);
  cv::Mat dst;
  cv::Size kernel_size = cv::Size(ksize, ksize);
  cv::GaussianBlur(src, dst, kernel_size, 0);
  // cv::bilateralFilter(src, dst, ksize, 75, 75);
  Image dst_img = mat_to_image(dst);
  return dst_img;
}