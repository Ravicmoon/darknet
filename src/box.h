#pragma once
#include <vector>

#include "libapi.h"

typedef enum
{
  IOU,
  GIOU,
  MSE,
  DIOU,
  CIOU
} IOU_LOSS;

typedef enum
{
  DEFAULT_NMS,
  GREEDY_NMS,
  DIOU_NMS,
  CORNERS_NMS
} NMS_KIND;

typedef struct DxRep
{
  float dt, db, dl, dr;
} DxRep;

typedef struct Ious
{
  float iou, giou, diou, ciou;
  DxRep dx_iou;
  DxRep dx_giou;
} Ious;

class LIB_API Box
{
 public:
  Box();
  Box(float _x, float _y, float _w, float _h);
  Box(float const* f, int stride = 1);

  static float Overlap(float x1, float w1, float x2, float w2);
  static float Intersect(Box const& b1, Box const& b2);
  static float Union(Box const& b1, Box const& b2);
  static float Iou(Box const& b1, Box const& b2);
  static float Ciou(Box const& b1, Box const& b2);
  static float Diou(Box const& b1, Box const& b2, float beta = 0.6);
  static float Giou(Box const& b1, Box const& b2);
  static float Rmse(Box const& b1, Box const& b2);
  static float Iou(Box const& b1, Box const& b2, IOU_LOSS iou_type);
  static DxRep DxIou(Box pred, Box gt, IOU_LOSS iou_type);

 public:
  class AbsBox
  {
   public:
    AbsBox(Box const& b);                  // Box to AbsBox
    AbsBox(Box const& b1, Box const& b2);  // Minimum encompassing box

   public:
    float left, right, top, bottom;
  };

 public:
  float x, y, w, h;
};

typedef struct Detection
{
  Box bbox;
  int classes;
  float* prob;
  float* mask;
  float objectness;
  int sort_class;

  // Gaussian YOLOv3
  // tx, ty, tw, th uncertainty
  float* uc;

  // bit-0: center
  // bit-1: top-left
  // bit-2: bottom-right
  int points;
} Detection;

typedef struct MostProbDet
{
  Box bbox;
  int cid;
  float prob;
} MostProbDet;

LIB_API void NmsSort(Detection* dets, int total, int classes, float thresh);
LIB_API void DiouNmsSort(Detection* dets, int total, int classes, float thresh,
    NMS_KIND nms_kind, float beta);

LIB_API std::vector<MostProbDet> GetMostProbDets(Detection* dets, int num_dets);
