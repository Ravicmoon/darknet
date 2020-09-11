#include "visualize.h"

#include <algorithm>

#include "utils.h"

std::vector<cv::Scalar> palette = {
    CV_RGB(255, 0, 0),
    CV_RGB(0, 255, 0),
    CV_RGB(0, 0, 255),
    CV_RGB(255, 255, 0),
    CV_RGB(0, 255, 255),
    CV_RGB(255, 0, 255),
    CV_RGB(255, 255, 255),
    CV_RGB(128, 0, 0),
    CV_RGB(0, 128, 0),
    CV_RGB(0, 0, 128),
    CV_RGB(128, 128, 0),
    CV_RGB(0, 128, 128),
    CV_RGB(128, 0, 128),
    CV_RGB(128, 128, 128),
};

cv::Scalar GetRandColor(int idx) { return palette[idx % palette.size()]; }

void Mat2Image(cv::Mat const& mat, Image* image)
{
  int w = mat.cols;
  int h = mat.rows;
  int c = mat.channels();

  if (image->data == nullptr)
  {
    image->w = w;
    image->h = h;
    image->c = c;
    image->data = new float[h * w * c];
  }

  unsigned char* data = (unsigned char*)mat.data;
  size_t step = mat.step;

  for (int y = 0; y < h; y++)
  {
    for (int k = 0; k < c; k++)
    {
      for (int x = 0; x < w; x++)
      {
        image->data[k * w * h + y * w + x] =
            data[y * step + x * c + k] / 255.0f;
      }
    }
  }
}

void DrawYoloDetections(
    cv::Mat& img, std::vector<MostProbDet> const& dets, Metadata const& md)
{
  std::vector<std::string> name_list = md.NameList();

  int font = cv::HersheyFonts::FONT_HERSHEY_COMPLEX_SMALL;
  char tag[128];

  for (size_t i = 0; i < dets.size(); i++)
  {
    Box::AbsBox b(dets[i].bbox);
    float left = b.left * img.cols;
    float right = b.right * img.cols;
    float top = b.top * img.rows;
    float bottom = b.bottom * img.rows;

    int cid = dets[i].cid;
    float prob = dets[i].prob;

    sprintf(tag, "%s(%2.0f%%)", name_list[cid].c_str(), prob * 100);

    int baseline = 0;
    cv::Size text_size = cv::getTextSize(tag, font, 1, 1, &baseline);

    cv::Point2f pt1(left, top);
    cv::Point2f pt2(right, bottom);
    cv::Point2f pt_text(left, top - baseline / 2);
    cv::Point2f pt_text_bg1(left, top - baseline - text_size.height);
    cv::Point2f pt_text_bg2(left + text_size.width, top);

    cv::Scalar color = GetRandColor(cid);

    int width = max_val_cmp(1, img.cols / 640);

    cv::rectangle(img, pt1, pt2, color, width);
    cv::rectangle(img, pt_text_bg1, pt_text_bg2, color, -1);
    cv::rectangle(img, pt_text_bg1, pt_text_bg2, color, width);
    cv::putText(img, tag, pt_text, font, 1, CV_RGB(0, 0, 0), 1, cv::LINE_AA);
  }
}

void DrawYoloTrackings(
    cv::Mat& img, std::vector<yc::Track*> const& tracks, Metadata const& md)
{
  std::vector<std::string> name_list = md.NameList();

  int font = cv::HersheyFonts::FONT_HERSHEY_COMPLEX_SMALL;
  char tag[128];

  for (size_t i = 0; i < tracks.size(); i++)
  {
    Box::AbsBox b(tracks[i]->GetBox());
    float left = b.left * img.cols;
    float right = b.right * img.cols;
    float top = b.top * img.rows;
    float bottom = b.bottom * img.rows;

    int cid = tracks[i]->GetClassId();
    int label = tracks[i]->GetLabel();
    float prob = tracks[i]->GetClassProb();

    sprintf(tag, "%s(%d,%2.0f%%)", name_list[cid].c_str(), label, prob * 100);

    int baseline = 0;
    cv::Size text_size = cv::getTextSize(tag, font, 0.5, 1, &baseline);

    cv::Point2f pt1(left, top);
    cv::Point2f pt2(right, bottom);
    cv::Point2f pt_text(left, top - baseline / 2);
    cv::Point2f pt_text_bg1(left, top - baseline - text_size.height);
    cv::Point2f pt_text_bg2(left + text_size.width, top);

    cv::Scalar color = GetRandColor(label);

    int width = std::max(1, img.cols / 640);
    if (tracks[i]->GetStatus() == yc::STATIONARY)
      width *= 2;

    cv::rectangle(img, pt1, pt2, color, width);
    cv::rectangle(img, pt_text_bg1, pt_text_bg2, color, -1);
    cv::rectangle(img, pt_text_bg1, pt_text_bg2, color, width);
    cv::putText(img, tag, pt_text, font, 0.5, CV_RGB(0, 0, 0), 1, cv::LINE_AA);
  }
}

void DrawProcTime(cv::Mat& img, int64_t millisec)
{
  std::stringstream ss;
  ss << "Proc time: " << millisec << " ms";
  cv::putText(img, ss.str(), cv::Point(10, 25), cv::FONT_HERSHEY_COMPLEX_SMALL,
      1.0, CV_RGB(255, 255, 255), 4);
  cv::putText(img, ss.str(), cv::Point(10, 25), cv::FONT_HERSHEY_COMPLEX_SMALL,
      1.0, CV_RGB(255, 0, 0), 1);
}

void DrawFrameInfo(cv::Mat& img, int64_t curr_frame, int64_t max_frame)
{
  std::stringstream ss;
  ss << "Frame: " << curr_frame << '/' << max_frame;
  cv::putText(img, ss.str(), cv::Point(10, 50), cv::FONT_HERSHEY_COMPLEX_SMALL,
      1.0, CV_RGB(255, 255, 255), 4);
  cv::putText(img, ss.str(), cv::Point(10, 50), cv::FONT_HERSHEY_COMPLEX_SMALL,
      1.0, CV_RGB(255, 0, 0), 1);
}

int const kMargin = 60;
int const kFont = cv::FONT_HERSHEY_COMPLEX_SMALL;
float const kFontSz = 0.7;

cv::Scalar const kBlue = CV_RGB(0, 0, 255);
cv::Scalar const kRed = CV_RGB(255, 0, 0);
cv::Scalar const kBlack = CV_RGB(0, 0, 0);
cv::Scalar const kWhite = CV_RGB(255, 255, 255);
cv::Scalar const kMajor = CV_RGB(128, 128, 128);
cv::Scalar const kMinor = CV_RGB(224, 224, 224);

cv::Mat DrawLossGraphBg(
    int max_iter, float max_loss, int num_lines, int img_size)
{
  int draw_size = img_size - 2 * kMargin;

  cv::Mat img(img_size, img_size, CV_8UC3, kWhite);
  cv::Point pt1, pt2, pt_txt;

  char buff[128];

  // vertical lines
  pt1.x = kMargin;
  pt2.x = kMargin + draw_size;

  for (int i = 0; i <= num_lines; ++i)
  {
    pt1.y = pt2.y = pt_txt.y = kMargin + (float)i * draw_size / num_lines;
    cv::line(img, pt1, pt2, kMinor, 1, 8, 0);
    if (i % 10 == 0)
    {
      sprintf(buff, "%2.1f", max_loss * (num_lines - i) / num_lines);

      cv::Size txt_sz = cv::getTextSize(buff, kFont, kFontSz, 1, nullptr);
      pt_txt.x = kMargin - txt_sz.width - 5;
      pt_txt.y += txt_sz.height / 2;

      cv::putText(img, buff, pt_txt, kFont, kFontSz, kBlue, 1, cv::LINE_AA);

      sprintf(buff, "%d%%", 100 * (num_lines - i) / num_lines);
      pt_txt.x = kMargin + draw_size + 5;

      cv::putText(img, buff, pt_txt, kFont, kFontSz, kRed, 1, cv::LINE_AA);

      cv::line(img, pt1, pt2, kMajor, 1, 8, 0);
    }
  }

  // horizontal lines
  pt1.y = kMargin;
  pt2.y = kMargin + draw_size;
  pt_txt.y = kMargin + draw_size + 20;

  for (int i = 0; i <= num_lines; ++i)
  {
    pt1.x = pt2.x = pt_txt.x = kMargin + (float)i * draw_size / num_lines;
    cv::line(img, pt1, pt2, kMinor, 1, 8, 0);
    if (i % 10 == 0)
    {
      int major_tick = max_iter * i / num_lines;
      if (major_tick > 1e6)
        sprintf(buff, "%.1lfM", major_tick / 1e6);
      else if (major_tick > 1e3)
        sprintf(buff, "%.1lfK", major_tick / 1e3);
      else
        sprintf(buff, "%d", major_tick);

      cv::Size txt_sz = cv::getTextSize(buff, kFont, kFontSz, 1, nullptr);
      pt_txt.x -= txt_sz.width / 2;

      cv::putText(img, buff, pt_txt, kFont, kFontSz, kBlack, 1, cv::LINE_AA);
      cv::line(img, pt1, pt2, kMajor, 1, 8, 0);
    }
  }

  cv::putText(img, "loss", cv::Point(10, kMargin / 2), kFont, kFontSz, kBlack,
      1, cv::LINE_AA);
  cv::putText(img, "# iter",
      cv::Point(kMargin + draw_size / 2, kMargin + draw_size + 35), kFont,
      kFontSz, kBlack, 1, cv::LINE_AA);

  return img;
}

void DrawLossGraph(cv::Mat const& bg, std::vector<int> const& iter,
    std::vector<float> const& avg_loss, std::vector<int> const& iter_map,
    std::vector<float> const& map, int max_iter, float max_loss,
    double time_remaining)
{
  if (iter.size() != avg_loss.size() || iter_map.size() != map.size())
    return;

  cv::Mat img;
  bg.copyTo(img);

  int draw_size = img.rows - 2 * kMargin;

  cv::Point2f pt1, offset(kMargin, kMargin);
  for (size_t i = 0; i < avg_loss.size(); i++)
  {
    pt1.x = draw_size * (float)iter[i] / max_iter;
    pt1.y = max_val_cmp(1.0f, draw_size * (1 - avg_loss[i] / max_loss));
    cv::drawMarker(img, pt1 + offset, kBlue, cv::MARKER_CROSS, 2);
  }

  cv::Point2f pt2;
  float max_map = 0.0f, max_iter_map = 0.0f;
  for (size_t i = 0; i < map.size(); i++)
  {
    pt1.x = draw_size * (float)iter_map[i] / max_iter;
    pt1.y = draw_size * (1.0f - map[i]);
    pt1 += offset;

    if (i != map.size() - 1)
    {
      pt2.x = draw_size * (float)iter_map[i + 1] / max_iter;
      pt2.y = draw_size * (1.0f - map[i + 1]);
      pt2 += offset;

      cv::line(img, pt1, pt2, kRed);
    }

    if (max_map < map[i])
    {
      max_map = map[i];
      max_iter_map = iter_map[i];
    }
  }

  // draw maximum mAP
  char buff[128];
  if (abs(max_map) > FLT_EPSILON)
  {
    pt1.x = draw_size * max_iter_map / max_iter;
    pt1.y = draw_size * (1.0f - max_map);
    pt1 += offset;

    sprintf(buff, "%2.0f%% ", max_map * 100.0f);
    cv::putText(img, buff, pt1, kFont, kFontSz, kWhite, 3, cv::LINE_AA);
    cv::putText(img, buff, pt1, kFont, kFontSz, kRed, 1, cv::LINE_AA);
  }

  sprintf(buff, "iter: %d  avg loss: %2.2f  time remaining: %2.2lf hrs",
      iter.back(), avg_loss.back(), time_remaining);
  cv::putText(img, buff, cv::Point(kMargin, kMargin + draw_size + 50), kFont,
      kFontSz, kBlue, 1, cv::LINE_AA);

  cv::imshow("loss_graph", img);
  cv::moveWindow("loss_graph", 0, 0);
  cv::resizeWindow("loss_graph", img.size());
  int key = cv::waitKey(1);

  if (key == 's' || iter.back() == (max_iter - 1) || iter.back() % 10 == 0)
    cv::imwrite("chart.png", img);
}