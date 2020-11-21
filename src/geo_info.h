
#include <ctime>
#include <deque>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "box.h"
#include "track_manager.h"

namespace yc
{
typedef std::vector<cv::Point2f> Polygon;

Polygon PolygonIntersection(Polygon const& poly1, Polygon const& poly2);
bool IsInPolygon(Polygon const& poly, cv::Point2f pt);
float PolygonArea(Polygon const& poly);

struct Occ
{
  int label;
  int sframe;
  int eframe;
  time_t start;
  time_t end;
};

class PolyRegion
{
 public:
  PolyRegion(std::string name, Polygon const& poly);

  std::string Name() const;

  bool IsInPolygon(cv::Point2f pt) const;
  virtual void Draw(cv::Mat& img, char const* msg = nullptr) const;
  virtual void Proc(std::vector<yc::Track*>& tracks, void* data = nullptr);

 protected:
  std::string name_;
  Polygon poly_;
  Box bbox_;
};

struct HandoverInfo
{
  yc::Track* track;
  int duration;
};

class Handover : public PolyRegion
{
 public:
  Handover(std::string name, Polygon const& poly);

  virtual void Proc(std::vector<yc::Track*>& tracks, void* data = nullptr);

  static void Crosstalk(Handover* h1, Handover* h2);

 protected:
  void ManageQueue(std::deque<HandoverInfo>& q);
  void UniquePushBack(std::deque<HandoverInfo>& q, yc::Track* track);

  std::deque<HandoverInfo> enter_;
  std::deque<HandoverInfo> exit_;
};

class ParkingLot : public PolyRegion
{
 public:
  ParkingLot(std::string name, Polygon const& poly);

  virtual void Draw(cv::Mat& img, char const* msg = nullptr) const;
  virtual void Proc(std::vector<yc::Track*>& tracks, void* data = nullptr);

  // for performance evaluation
  void SaveHistory(std::string path);

 protected:
  Occ curr_occ_;
  std::vector<Occ> occupations_;
};

class GeoInfo
{
 public:
  ~GeoInfo();

  void Load(std::string xml_path);

  void Draw(cv::Mat& img) const;
  void Proc(std::vector<yc::Track*>& tracks, void* data = nullptr);

  int NumHandoverRegions() const;
  Handover* GetHandoverRegion(int idx);

  // for performance evaluation
  void SaveParkingLotHistory(std::string path);

 private:
  std::vector<ParkingLot*> parking_lots;
  std::vector<Handover*> handovers;
};
}  // namespace yc