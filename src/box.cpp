#include "box.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

#ifndef M_PI
#define M_PI 3.141592
#endif

#define SQUARE(x) ((x) * (x))

Box::Box() : x(0.0f), y(0.0f), w(0.0f), h(0.0f) {}

Box::Box(float _x, float _y, float _w, float _h) : x(_x), y(_y), w(_w), h(_h) {}

Box::Box(float const* f, int stride)
    : x(f[0]), y(f[1 * stride]), w(f[2 * stride]), h(f[3 * stride])
{
}

float Box::Overlap(float x1, float w1, float x2, float w2)
{
  float l1 = x1 - w1 / 2;
  float l2 = x2 - w2 / 2;
  float left = l1 > l2 ? l1 : l2;
  float r1 = x1 + w1 / 2;
  float r2 = x2 + w2 / 2;
  float right = r1 < r2 ? r1 : r2;

  return right - left;
}

float Box::Intersect(Box const& b1, Box const& b2)
{
  float w = Overlap(b1.x, b1.w, b2.x, b2.w);
  float h = Overlap(b1.y, b1.h, b2.y, b2.h);

  if (w < 0 || h < 0)
    return 0;
  else
    return w * h;
}

float Box::Union(Box const& b1, Box const& b2)
{
  return b1.w * b1.h + b2.w * b2.h - Intersect(b1, b2);
}

float Box::Iou(Box const& b1, Box const& b2)
{
  float I = Intersect(b1, b2);
  float U = Union(b1, b2);
  if (abs(I) < FLT_EPSILON || abs(U) < FLT_EPSILON)
    return 0;
  else
    return I / U;
}

float Box::Ciou(Box const& b1, Box const& b2)
{
  AbsBox ab(b1, b2);

  float w = ab.right - ab.left;
  float h = ab.bottom - ab.top;
  float c = w * w + h * h;
  float iou = Iou(b1, b2);
  if (abs(c) < FLT_EPSILON)
    return iou;

  float u = SQUARE(b1.x - b2.x) + SQUARE(b1.y - b2.y);
  float d = u / c;
  float ar_gt = b2.w / b2.h;
  float ar_pred = b1.w / b1.h;
  float ar_loss = 4 / SQUARE(M_PI) * SQUARE(atan(ar_gt) - atan(ar_pred));
  float alpha = ar_loss / (1 - iou + ar_loss + 0.000001);
  float ciou_term = d + alpha * ar_loss;

  return iou - ciou_term;
}

float Box::Diou(Box const& b1, Box const& b2, float beta)
{
  AbsBox ab(b1, b2);

  float w = ab.right - ab.left;
  float h = ab.bottom - ab.top;
  float c = w * w + h * h;
  float iou = Iou(b1, b2);
  if (abs(c) < FLT_EPSILON)
    return iou;

  float d = SQUARE(b1.x - b2.x) + SQUARE(b1.y - b2.y);
  float diou_term = pow(d / c, beta);

  return iou - diou_term;
}

float Box::Giou(Box const& b1, Box const& b2)
{
  AbsBox ab(b1, b2);

  float w = ab.right - ab.left;
  float h = ab.bottom - ab.top;
  float c = w * h;
  float iou = Iou(b1, b2);
  if (abs(c) < FLT_EPSILON)
    return iou;

  float u = Union(b1, b2);
  float giou_term = (c - u) / c;

  return iou - giou_term;
}

float Box::Rmse(Box const& b1, Box const& b2)
{
  return sqrt(SQUARE(b1.x - b2.x) + SQUARE(b1.y - b2.y) + SQUARE(b1.w - b2.w) +
              SQUARE(b1.h - b2.h));
}

float Box::Iou(Box const& b1, Box const& b2, IOU_LOSS iou_type)
{
  switch (iou_type)
  {
    case GIOU:
      return Giou(b1, b2);
    case MSE:
      return Rmse(b1, b2);
    case DIOU:
      return Diou(b1, b2);
    case CIOU:
      return Ciou(b1, b2);
    default:
      return Iou(b1, b2);
  }
}

DxRep Box::DxIou(Box pred, Box gt, IOU_LOSS iou_type)
{
  Box::AbsBox ab_pred(pred);
  float pred_t = fmin(ab_pred.top, ab_pred.bottom);
  float pred_b = fmax(ab_pred.top, ab_pred.bottom);
  float pred_l = fmin(ab_pred.left, ab_pred.right);
  float pred_r = fmax(ab_pred.left, ab_pred.right);
  Box::AbsBox ab_gt(gt);

  float X = (pred_b - pred_t) * (pred_r - pred_l);
  float Xhat = (ab_gt.bottom - ab_gt.top) * (ab_gt.right - ab_gt.left);
  float Ih = fmin(pred_b, ab_gt.bottom) - fmax(pred_t, ab_gt.top);
  float Iw = fmin(pred_r, ab_gt.right) - fmax(pred_l, ab_gt.left);
  float I = Iw * Ih;
  float U = X + Xhat - I;
  float S = SQUARE(pred.x - gt.x) + SQUARE(pred.y - gt.y);
  float giou_Cw = fmax(pred_r, ab_gt.right) - fmin(pred_l, ab_gt.left);
  float giou_Ch = fmax(pred_b, ab_gt.bottom) - fmin(pred_t, ab_gt.top);
  float giou_C = giou_Cw * giou_Ch;

  // Partial Derivatives, derivatives
  float dX_wrt_t = -1 * (pred_r - pred_l);
  float dX_wrt_b = pred_r - pred_l;
  float dX_wrt_l = -1 * (pred_b - pred_t);
  float dX_wrt_r = pred_b - pred_t;

  // gradient of I min/max in IoU calc (prediction)
  float dI_wrt_t = pred_t > ab_gt.top ? (-1 * Iw) : 0;
  float dI_wrt_b = pred_b < ab_gt.bottom ? Iw : 0;
  float dI_wrt_l = pred_l > ab_gt.left ? (-1 * Ih) : 0;
  float dI_wrt_r = pred_r < ab_gt.right ? Ih : 0;

  // derivative of U with regard to x
  float dU_wrt_t = dX_wrt_t - dI_wrt_t;
  float dU_wrt_b = dX_wrt_b - dI_wrt_b;
  float dU_wrt_l = dX_wrt_l - dI_wrt_l;
  float dU_wrt_r = dX_wrt_r - dI_wrt_r;

  // gradient of C min/max in IoU calc (prediction)
  float dC_wrt_t = pred_t < ab_gt.top ? (-1 * giou_Cw) : 0;
  float dC_wrt_b = pred_b > ab_gt.bottom ? giou_Cw : 0;
  float dC_wrt_l = pred_l < ab_gt.left ? (-1 * giou_Ch) : 0;
  float dC_wrt_r = pred_r > ab_gt.right ? giou_Ch : 0;

  float p_dt = 0;
  float p_db = 0;
  float p_dl = 0;
  float p_dr = 0;
  if (U > 0)
  {
    p_dt = ((U * dI_wrt_t) - (I * dU_wrt_t)) / (U * U);
    p_db = ((U * dI_wrt_b) - (I * dU_wrt_b)) / (U * U);
    p_dl = ((U * dI_wrt_l) - (I * dU_wrt_l)) / (U * U);
    p_dr = ((U * dI_wrt_r) - (I * dU_wrt_r)) / (U * U);
  }
  // apply grad from prediction min/max for correct corner selection
  p_dt = ab_pred.top < ab_pred.bottom ? p_dt : p_db;
  p_db = ab_pred.top < ab_pred.bottom ? p_db : p_dt;
  p_dl = ab_pred.left < ab_pred.right ? p_dl : p_dr;
  p_dr = ab_pred.left < ab_pred.right ? p_dr : p_dl;

  if (iou_type == GIOU)
  {
    if (giou_C > 0)
    {
      // apply "C" term from gIOU
      p_dt += ((giou_C * dU_wrt_t) - (U * dC_wrt_t)) / (giou_C * giou_C);
      p_db += ((giou_C * dU_wrt_b) - (U * dC_wrt_b)) / (giou_C * giou_C);
      p_dl += ((giou_C * dU_wrt_l) - (U * dC_wrt_l)) / (giou_C * giou_C);
      p_dr += ((giou_C * dU_wrt_r) - (U * dC_wrt_r)) / (giou_C * giou_C);
    }
    if (Iw <= 0 || Ih <= 0)
    {
      p_dt = ((giou_C * dU_wrt_t) - (U * dC_wrt_t)) / (giou_C * giou_C);
      p_db = ((giou_C * dU_wrt_b) - (U * dC_wrt_b)) / (giou_C * giou_C);
      p_dl = ((giou_C * dU_wrt_l) - (U * dC_wrt_l)) / (giou_C * giou_C);
      p_dr = ((giou_C * dU_wrt_r) - (U * dC_wrt_r)) / (giou_C * giou_C);
    }
  }

  float Ct = fmin(pred.y - pred.h / 2, gt.y - gt.h / 2);
  float Cb = fmax(pred.y + pred.h / 2, gt.y + gt.h / 2);
  float Cl = fmin(pred.x - pred.w / 2, gt.x - gt.w / 2);
  float Cr = fmax(pred.x + pred.w / 2, gt.x + gt.w / 2);
  float Cw = Cr - Cl;
  float Ch = Cb - Ct;
  float C = SQUARE(Cw) + SQUARE(Ch);

  float dCt_dx = 0;
  float dCt_dy = pred_t < ab_gt.top ? 1 : 0;
  float dCt_dw = 0;
  float dCt_dh = pred_t < ab_gt.top ? -0.5 : 0;

  float dCb_dx = 0;
  float dCb_dy = pred_b > ab_gt.bottom ? 1 : 0;
  float dCb_dw = 0;
  float dCb_dh = pred_b > ab_gt.bottom ? 0.5 : 0;

  float dCl_dx = pred_l < ab_gt.left ? 1 : 0;
  float dCl_dy = 0;
  float dCl_dw = pred_l < ab_gt.left ? -0.5 : 0;
  float dCl_dh = 0;

  float dCr_dx = pred_r > ab_gt.right ? 1 : 0;
  float dCr_dy = 0;
  float dCr_dw = pred_r > ab_gt.right ? 0.5 : 0;
  float dCr_dh = 0;

  float dCw_dx = dCr_dx - dCl_dx;
  float dCw_dy = dCr_dy - dCl_dy;
  float dCw_dw = dCr_dw - dCl_dw;
  float dCw_dh = dCr_dh - dCl_dh;

  float dCh_dx = dCb_dx - dCt_dx;
  float dCh_dy = dCb_dy - dCt_dy;
  float dCh_dw = dCb_dw - dCt_dw;
  float dCh_dh = dCb_dh - dCt_dh;

  // Final IoU loss (prediction)
  // (negative of IoU gradient, we want the negative loss)

  // p_dx, p_dy, p_dw and p_dh are the gradient of IoU or GIoU.
  float p_dx = p_dl + p_dr;
  float p_dy = p_dt + p_db;
  // For dw and dh, we do not divide them by 2.
  float p_dw = (p_dr - p_dl);
  float p_dh = (p_db - p_dt);

  // https://github.com/Zzh-tju/DIoU-darknet
  // https://arxiv.org/abs/1911.08287
  if (iou_type == DIOU)
  {
    if (C > 0)
    {
      p_dx +=
          (2 * (gt.x - pred.x) * C - (2 * Cw * dCw_dx + 2 * Ch * dCh_dx) * S) /
          SQUARE(C);
      p_dy +=
          (2 * (gt.y - pred.y) * C - (2 * Cw * dCw_dy + 2 * Ch * dCh_dy) * S) /
          SQUARE(C);
      p_dw += (2 * Cw * dCw_dw + 2 * Ch * dCh_dw) * S / SQUARE(C);
      p_dh += (2 * Cw * dCw_dh + 2 * Ch * dCh_dh) * S / SQUARE(C);
    }
    if (Iw <= 0 || Ih <= 0)
    {
      p_dx =
          (2 * (gt.x - pred.x) * C - (2 * Cw * dCw_dx + 2 * Ch * dCh_dx) * S) /
          SQUARE(C);
      p_dy =
          (2 * (gt.y - pred.y) * C - (2 * Cw * dCw_dy + 2 * Ch * dCh_dy) * S) /
          SQUARE(C);
      p_dw = (2 * Cw * dCw_dw + 2 * Ch * dCh_dw) * S / SQUARE(C);
      p_dh = (2 * Cw * dCw_dh + 2 * Ch * dCh_dh) * S / SQUARE(C);
    }
  }

  // The following codes are calculating the gradient of ciou.
  if (iou_type == CIOU)
  {
    float ar_gt = gt.w / gt.h;
    float ar_pred = pred.w / pred.h;
    float ar_loss = 4 / SQUARE(M_PI) * SQUARE(atan(ar_gt) - atan(ar_pred));
    float alpha = ar_loss / (1 - I / U + ar_loss + 0.000001);
    float ar_dw = 8 / SQUARE(M_PI) * (atan(ar_gt) - atan(ar_pred)) * pred.h;
    float ar_dh = -8 / SQUARE(M_PI) * (atan(ar_gt) - atan(ar_pred)) * pred.w;

    if (C > 0)
    {
      p_dx +=
          (2 * (gt.x - pred.x) * C - (2 * Cw * dCw_dx + 2 * Ch * dCh_dx) * S) /
          SQUARE(C);
      p_dy +=
          (2 * (gt.y - pred.y) * C - (2 * Cw * dCw_dy + 2 * Ch * dCh_dy) * S) /
          SQUARE(C);
      p_dw +=
          (2 * Cw * dCw_dw + 2 * Ch * dCh_dw) * S / SQUARE(C) + alpha * ar_dw;
      p_dh +=
          (2 * Cw * dCw_dh + 2 * Ch * dCh_dh) * S / SQUARE(C) + alpha * ar_dh;
    }

    if (Iw <= 0 || Ih <= 0)
    {
      p_dx =
          (2 * (gt.x - pred.x) * C - (2 * Cw * dCw_dx + 2 * Ch * dCh_dx) * S) /
          SQUARE(C);
      p_dy =
          (2 * (gt.y - pred.y) * C - (2 * Cw * dCw_dy + 2 * Ch * dCh_dy) * S) /
          SQUARE(C);
      p_dw =
          (2 * Cw * dCw_dw + 2 * Ch * dCh_dw) * S / SQUARE(C) + alpha * ar_dw;
      p_dh =
          (2 * Cw * dCw_dh + 2 * Ch * dCh_dh) * S / SQUARE(C) + alpha * ar_dh;
    }
  }

  // We follow the original code released from GDarknet.
  // So in yolo_layer.c, dt, db, dl, dr are already dx, dy, dw, dh.
  return DxRep{p_dx, p_dy, p_dw, p_dh};
}

Box::AbsBox::AbsBox(Box const& b)
    : left(b.x - b.w / 2.0f),
      right(b.x + b.w / 2.0f),
      top(b.y - b.h / 2.0f),
      bottom(b.y + b.h / 2.0f)
{
}

Box::AbsBox::AbsBox(Box const& b1, Box const& b2)
{
  AbsBox ab1(b1);
  AbsBox ab2(b2);

  left = fmin(ab1.left, ab2.left);
  right = fmax(ab1.right, ab2.right);
  top = fmin(ab1.top, ab2.top);
  bottom = fmax(ab1.bottom, ab2.bottom);
}

int NmsComparator(const void* pa, const void* pb)
{
  Detection* a = (Detection*)pa;
  Detection* b = (Detection*)pb;

  float diff = 0;
  if (b->sort_class >= 0)
    diff = a->prob[b->sort_class] - b->prob[b->sort_class];
  else
    diff = a->objectness - b->objectness;

  if (diff < 0)
    return 1;
  else if (diff > 0)
    return -1;
  else
    return 0;
}

void NmsSort(Detection* dets, int total, int classes, float thresh)
{
  int k = total - 1;
  for (int i = 0; i <= k; ++i)
  {
    if (abs(dets[i].objectness) < FLT_EPSILON)
    {
      Detection swap = dets[i];
      dets[i] = dets[k];
      dets[k] = swap;
      --k;
      --i;
    }
  }
  total = k + 1;

  for (k = 0; k < classes; ++k)
  {
    for (int i = 0; i < total; ++i)
    {
      dets[i].sort_class = k;
    }
    qsort(dets, total, sizeof(Detection), NmsComparator);
    for (int i = 0; i < total; ++i)
    {
      if (abs(dets[i].prob[k]) < FLT_EPSILON)
        continue;

      Box a = dets[i].bbox;
      for (int j = i + 1; j < total; ++j)
      {
        Box b = dets[j].bbox;
        if (Box::Iou(a, b) > thresh)
          dets[j].prob[k] = 0.0f;
      }
    }
  }
}

// https://github.com/Zzh-tju/DIoU-darknet
// https://arxiv.org/abs/1911.08287
void DiouNmsSort(Detection* dets, int total, int classes, float thresh,
    NMS_KIND nms_kind, float beta)
{
  int k = total - 1;
  for (int i = 0; i <= k; ++i)
  {
    if (dets[i].objectness == 0)
    {
      Detection swap = dets[i];
      dets[i] = dets[k];
      dets[k] = swap;
      --k;
      --i;
    }
  }
  total = k + 1;

  for (k = 0; k < classes; ++k)
  {
    for (int i = 0; i < total; ++i)
    {
      dets[i].sort_class = k;
    }
    qsort(dets, total, sizeof(Detection), NmsComparator);
    for (int i = 0; i < total; ++i)
    {
      if (abs(dets[i].prob[k]) < FLT_EPSILON)
        continue;

      Box a = dets[i].bbox;
      for (int j = i + 1; j < total; ++j)
      {
        Box b = dets[j].bbox;
        if (Box::Iou(a, b) > thresh && nms_kind == CORNERS_NMS)
        {
          dets[j].prob[k] = 0.0f;
        }
        else if (Box::Iou(a, b) > thresh && nms_kind == GREEDY_NMS)
        {
          dets[j].prob[k] = 0.0f;
        }
        else
        {
          if (Box::Diou(a, b, beta) > thresh && nms_kind == DIOU_NMS)
            dets[j].prob[k] = 0.0f;
        }
      }
    }
  }
}

LIB_API std::vector<MostProbDet> GetMostProbDets(Detection* dets, int num_dets)
{
  std::vector<MostProbDet> most_prob_dets;
  for (int i = 0; i < num_dets; i++)
  {
    int cid = -1;
    float max_prob = 0.0f;
    for (int j = 0; j < dets[i].classes; j++)
    {
      if (dets[i].prob[j] > max_prob)
      {
        cid = j;
        max_prob = dets[i].prob[j];
      }
    }

    if (cid != -1)
    {
      MostProbDet mpd;
      mpd.bbox = dets[i].bbox;
      mpd.cid = cid;
      mpd.prob = max_prob;
      most_prob_dets.push_back(mpd);
    }
  }

  return most_prob_dets;
}