#include "geo_info.h"

#include <tinyxml2.h>

#include "utils.h"

namespace yc
{
int const kFont = cv::FONT_HERSHEY_COMPLEX_SMALL;
float const kFontSz = 0.7f;

cv::Scalar const kRed = CV_RGB(255, 0, 0);
cv::Scalar const kWhite = CV_RGB(255, 255, 255);

bool GetIntersectLineSeg(cv::Point2f p0, cv::Point2f p1, cv::Point2f p2,
    cv::Point p3, cv::Point2f& i)
{
  cv::Point2f s1(p1.x - p0.x, p1.y - p0.y);
  cv::Point2f s2(p3.x - p2.x, p3.y - p2.y);

  float s = (-s1.y * (p0.x - p2.x) + s1.x * (p0.y - p2.y)) /
            (-s2.x * s1.y + s1.x * s2.y);
  float t = (s2.x * (p0.y - p2.y) - s2.y * (p0.x - p2.x)) /
            (-s2.x * s1.y + s1.x * s2.y);

  if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
  {
    // Collision detected
    i.x = p0.x + (t * s1.x);
    i.y = p0.y + (t * s1.y);

    return true;
  }

  return false;  // No collision
}

// not working correctly, need to fix this
Polygon PolygonIntersection(Polygon const& p1, Polygon const& p2)
{
  Polygon ret;
  for (size_t i = 0; i < p1.size(); i++)
  {
    for (size_t j = 0; j < p2.size(); j++)
    {
      size_t ii = (i + 1) % p1.size();
      size_t jj = (j + 1) % p2.size();
      cv::Point2f pt_i;
      if (GetIntersectLineSeg(p1[i], p1[ii], p2[j], p2[jj], pt_i))
        ret.push_back(pt_i);
    }
  }

  for (size_t i = 0; i < ret.size(); i++)
  {
    for (size_t j = 0; j < p1.size(); j++)
    {
      if (cv::norm(ret[i] - p1[j]) > FLT_EPSILON)
        ret.push_back(p1[j]);
    }
    for (size_t j = 0; j < p2.size(); j++)
    {
      if (cv::norm(ret[i] - p2[j]) > FLT_EPSILON)
        ret.push_back(p2[j]);
    }
  }

  return ret;
}

bool IsInPolygon(Polygon const& poly, cv::Point2f pt)
{
  bool is_inside = false;

  int j = (int)poly.size() - 1;
  for (int i = 0; i < (int)poly.size(); i++)
  {
    if ((poly[i].y > pt.y != poly[j].y > pt.y) &&
        (pt.x < poly[i].x + (poly[j].x - poly[i].x) * (pt.y - poly[i].y) /
                                (poly[j].y - poly[i].y)))
      is_inside = !is_inside;

    j = i;
  }

  return is_inside;
}

float PolygonArea(Polygon const& poly)
{
  if (poly.size() < 3)
    return 0.0f;

  float area = 0.0f;

  int j = (int)poly.size() - 1;
  for (int i = 0; i < (int)poly.size(); i++)
  {
    area += (poly[j].x + poly[i].x) * (poly[j].y - poly[i].y);
    j = i;
  }

  return abs(area / 2.0f);
}

PolyInfo::PolyInfo(std::string name, Polygon const& poly)
    : name_(name), poly_(poly)
{
  float x_min = FLT_MAX, x_max = 0;
  float y_min = FLT_MAX, y_max = 0;
  for (size_t i = 0; i < poly.size(); i++)
  {
    x_min = min_val_cmp(x_min, poly[i].x);
    x_max = max_val_cmp(x_max, poly[i].x);
    y_min = min_val_cmp(y_min, poly[i].y);
    y_max = max_val_cmp(y_max, poly[i].y);
  }

  bbox_.x = (x_min + x_max) / 2.0f;
  bbox_.y = (y_min + y_max) / 2.0f;
  bbox_.w = x_max - x_min;
  bbox_.h = y_max - y_min;
}

std::string PolyInfo::Name() const { return name_; }

bool PolyInfo::IsInPolygon(cv::Point2f pt) const
{
  return yc::IsInPolygon(poly_, pt);
}

void PolyInfo::Draw(cv::Mat& img, char const* msg) const
{
  int width = img.cols;
  int height = img.rows;

  // scale and draw polygon
  std::vector<cv::Point> scaled(poly_.size());
  for (size_t i = 0; i < scaled.size(); i++)
  {
    scaled[i].x = int(poly_[i].x * width + 0.5f);
    scaled[i].y = int(poly_[i].y * height + 0.5f);
  }

  cv::polylines(img, scaled, true, kRed);

  // draw message
  cv::Point2f center;
  for (size_t j = 0; j < poly_.size(); j++)
  {
    center.x += poly_[j].x;
    center.y += poly_[j].y;
  }
  center.x = center.x * width / poly_.size();
  center.y = center.y * height / poly_.size();

  if (msg == nullptr)
    msg = name_.c_str();

  cv::Size msg_sz = cv::getTextSize(msg, kFont, kFontSz, 1, nullptr);
  center.x -= msg_sz.width / 2;
  center.y += msg_sz.height / 2;

  cv::putText(img, msg, center, kFont, kFontSz, kWhite, 3, cv::LINE_AA);
  cv::putText(img, msg, center, kFont, kFontSz, kRed, 1, cv::LINE_AA);
}

void PolyInfo::Proc(std::vector<yc::Track*>& tracks) {}

Handover::Handover(std::string name, Polygon const& poly) : PolyInfo(name, poly)
{
}

void Handover::Proc(std::vector<yc::Track*>& tracks)
{
  for (size_t i = 0; i < tracks.size(); i++)
  {
    Box box = tracks[i]->GetBox();

    float area_i = Box::Intersect(bbox_, box);
    if (area_i / (box.w * box.h) > 0.5f)
    {
      if (tracks[i]->GetCount() < Track::GetFps() * 2)
      {
        if (!tracks[i]->GetEnterStatus())
          UniquePushBack(enter_, tracks[i]);
      }
      else
      {
        if (!tracks[i]->GetExitStatus())
          UniquePushBack(exit_, tracks[i]);
      }
    }
  }
}

void Handover::Crosstalk(Handover* h1, Handover* h2)
{
  if (!h1->exit_.empty() && !h2->enter_.empty())
  {
    std::string lp = h1->exit_.front()->GetLicensePlate();
    int label = h1->exit_.front()->GetLabel();
    if (label != -1)
    {
      h2->enter_.front()->SetLicensePlate(lp);
      h2->enter_.front()->SetLabel(label);
      h2->enter_.front()->SetEnterStatus(true);
      h1->exit_.front()->SetExitStatus(true);

      h1->exit_.pop_front();
      h2->enter_.pop_front();
    }
  }

  if (!h2->exit_.empty() && !h1->enter_.empty())
  {
    std::string lp = h2->exit_.front()->GetLicensePlate();
    int label = h2->exit_.front()->GetLabel();
    if (label != -1)
    {
      h1->enter_.front()->SetLicensePlate(lp);
      h1->enter_.front()->SetLabel(label);
      h1->enter_.front()->SetEnterStatus(true);
      h2->exit_.front()->SetExitStatus(true);

      h2->exit_.pop_front();
      h1->enter_.pop_front();
    }
  }
}

void Handover::UniquePushBack(std::deque<yc::Track*>& q, yc::Track* track)
{
  bool exist = false;
  for (size_t i = 0; i < q.size(); i++)
  {
    if (q[i] == track)
    {
      exist = true;
      break;
    }
  }

  if (!exist)
    q.push_back(track);
}

ParkingLot::ParkingLot(std::string name, Polygon const& poly)
    : PolyInfo(name, poly), curr_occ_(Occ())
{
}

void ParkingLot::Draw(cv::Mat& img, char const* msg) const
{
  char buff[256] = "";
  if (curr_occ_.start != 0)
  {
    time_t curr_time = time(nullptr);
    int diff = (int)difftime(curr_time, curr_occ_.start);

    int s = diff % 60;
    int m = diff / 60 % 60;
    int h = diff / 3600;

    sprintf(buff, "%02d:%02d:%02d", h, m, s);
  }

  PolyInfo::Draw(img, buff);
}

void ParkingLot::Proc(std::vector<yc::Track*>& tracks)
{
  bool matched = false;
  for (size_t i = 0; i < tracks.size(); i++)
  {
    Box bbox = tracks[i]->GetBox();
    cv::Point2f center(bbox.x, bbox.y);

    if (!IsInPolygon(center))
      continue;

    if (curr_occ_.start == 0 && tracks[i]->GetStatus() == STATIONARY)
    {
      curr_occ_.label = tracks[i]->GetLabel();
      time(&curr_occ_.start);
      matched = true;
    }
    else if (curr_occ_.start != 0 && curr_occ_.label == tracks[i]->GetLabel())
    {
      matched = true;
    }
  }

  if (!matched)
  {
    time(&curr_occ_.end);
    occupations_.push_back(curr_occ_);
    curr_occ_ = Occ();
  }
}

GeoInfo::~GeoInfo()
{
  for (size_t i = 0; i < parking_lots.size(); i++)
  {
    delete parking_lots[i];
  }

  for (size_t i = 0; i < handovers.size(); i++)
  {
    delete handovers[i];
  }
}

void GeoInfo::Load(std::string xml_path)
{
  tinyxml2::XMLDocument xml_doc;
  if (xml_doc.LoadFile(xml_path.c_str()) != tinyxml2::XML_SUCCESS)
    return;

  tinyxml2::XMLElement* root = xml_doc.FirstChildElement("polygons");
  tinyxml2::XMLElement* polygon = root->FirstChildElement("polygon");

  char buff[256];
  while (polygon != nullptr)
  {
    std::string name = polygon->FirstChildElement("name")->GetText();

    int num = polygon->FirstChildElement("num")->IntText();
    Polygon poly(num);
    for (size_t i = 0; i < poly.size(); i++)
    {
      sprintf(buff, "x%zd", i);
      float x = polygon->FirstChildElement(buff)->FloatText();

      sprintf(buff, "y%zd", i);
      float y = polygon->FirstChildElement(buff)->FloatText();

      poly[i] = cv::Point2f(x, y);
    }

    if (name.find_first_of("P") == 0)
      parking_lots.push_back(new ParkingLot(name, poly));
    else if (name == "HANDOVER")
      handovers.push_back(new Handover(name, poly));

    polygon = polygon->NextSiblingElement();
  }
}

void GeoInfo::Draw(cv::Mat& img) const
{
  for (size_t i = 0; i < parking_lots.size(); i++)
  {
    parking_lots[i]->Draw(img);
  }

  for (size_t i = 0; i < handovers.size(); i++)
  {
    handovers[i]->Draw(img);
  }
}

void GeoInfo::Proc(std::vector<yc::Track*>& tracks)
{
  for (size_t i = 0; i < parking_lots.size(); i++)
  {
    parking_lots[i]->Proc(tracks);
  }

  for (size_t i = 0; i < handovers.size(); i++)
  {
    handovers[i]->Proc(tracks);
  }
}

int GeoInfo::NumHandoverRegions() const { return (int)handovers.size(); }
Handover* GeoInfo::GetHandoverRegion(int idx) { return handovers[idx]; }
}  // namespace yc