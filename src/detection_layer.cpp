#include "detection_layer.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "activations.h"
#include "blas.h"
#include "box.h"
#include "dark_cuda.h"
#include "softmax_layer.h"
#include "utils.h"

void FillDetectionLayer(layer* l, int batch, int inputs, int n, int side,
    int classes, int coords, int rescore)
{
  l->type = DETECTION;
  l->n = n;
  l->batch = batch;
  l->inputs = inputs;
  l->classes = classes;
  l->coords = coords;
  l->rescore = rescore;
  l->side = side;
  l->w = side;
  l->h = side;
  assert(side * side * ((1 + l->coords) * l->n + l->classes) == inputs);
  l->cost = (float*)xcalloc(1, sizeof(float));
  l->outputs = l->inputs;
  l->truths = l->side * l->side * (1 + l->coords + l->classes);
  l->output = (float*)xcalloc(batch * l->outputs, sizeof(float));
  l->delta = (float*)xcalloc(batch * l->outputs, sizeof(float));

  l->forward = ForwardDetectionLayer;
  l->backward = BackwardDetectionLayer;
#ifdef GPU
  l->forward_gpu = ForwardDetectionLayerGpu;
  l->backward_gpu = BackwardDetectionLayerGpu;
  l->output_gpu = cuda_make_array(l->output, batch * l->outputs);
  l->delta_gpu = cuda_make_array(l->delta, batch * l->outputs);
#endif

  fprintf(stderr, "detection_layer\n");
  srand(time(0));
}

void ForwardDetectionLayer(layer* l, NetworkState state)
{
  int locations = l->side * l->side;
  int i, j;
  memcpy(l->output, state.input, l->outputs * l->batch * sizeof(float));

  int b;
  if (l->softmax)
  {
    for (b = 0; b < l->batch; ++b)
    {
      int index = b * l->inputs;
      for (i = 0; i < locations; ++i)
      {
        int offset = i * l->classes;
        softmax(l->output + index + offset, l->classes, 1,
            l->output + index + offset, 1);
      }
    }
  }
  if (state.train)
  {
    float avg_iou = 0;
    float avg_cat = 0;
    float avg_allcat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    *(l->cost) = 0;
    int size = l->inputs * l->batch;
    memset(l->delta, 0, size * sizeof(float));
    for (b = 0; b < l->batch; ++b)
    {
      int index = b * l->inputs;
      for (i = 0; i < locations; ++i)
      {
        int truth_index = (b * locations + i) * (1 + l->coords + l->classes);
        int is_obj = state.truth[truth_index];
        for (j = 0; j < l->n; ++j)
        {
          int p_index = index + locations * l->classes + i * l->n + j;
          l->delta[p_index] = l->noobject_scale * (0 - l->output[p_index]);
          *(l->cost) += l->noobject_scale * pow(l->output[p_index], 2);
          avg_anyobj += l->output[p_index];
        }

        int best_index = -1;
        float best_iou = 0;
        float best_rmse = 20;

        if (!is_obj)
        {
          continue;
        }

        int class_index = index + i * l->classes;
        for (j = 0; j < l->classes; ++j)
        {
          l->delta[class_index + j] =
              l->class_scale *
              (state.truth[truth_index + 1 + j] - l->output[class_index + j]);
          *(l->cost) += l->class_scale * pow(state.truth[truth_index + 1 + j] -
                                                 l->output[class_index + j],
                                             2);
          if (state.truth[truth_index + 1 + j])
            avg_cat += l->output[class_index + j];
          avg_allcat += l->output[class_index + j];
        }

        Box truth(state.truth + truth_index + 1 + l->classes);
        truth.x /= l->side;
        truth.y /= l->side;

        for (j = 0; j < l->n; ++j)
        {
          int box_index = index + locations * (l->classes + l->n) +
                          (i * l->n + j) * l->coords;
          Box out(l->output + box_index);
          out.x /= l->side;
          out.y /= l->side;

          if (l->sqrt)
          {
            out.w = out.w * out.w;
            out.h = out.h * out.h;
          }

          float iou = Box::Iou(out, truth);
          // iou = 0;
          float rmse = Box::Rmse(out, truth);
          if (best_iou > 0 || iou > 0)
          {
            if (iou > best_iou)
            {
              best_iou = iou;
              best_index = j;
            }
          }
          else
          {
            if (rmse < best_rmse)
            {
              best_rmse = rmse;
              best_index = j;
            }
          }
        }

        if (l->forced)
        {
          if (truth.w * truth.h < .1)
          {
            best_index = 1;
          }
          else
          {
            best_index = 0;
          }
        }
        if (l->random && *(state.net->seen) < 64000)
        {
          best_index = rand() % l->n;
        }

        int box_index = index + locations * (l->classes + l->n) +
                        (i * l->n + best_index) * l->coords;
        int tbox_index = truth_index + 1 + l->classes;

        Box out(l->output + box_index);
        out.x /= l->side;
        out.y /= l->side;
        if (l->sqrt)
        {
          out.w = out.w * out.w;
          out.h = out.h * out.h;
        }
        float iou = Box::Iou(out, truth);

        // printf("%d,", best_index);
        int p_index = index + locations * l->classes + i * l->n + best_index;
        *(l->cost) -= l->noobject_scale * pow(l->output[p_index], 2);
        *(l->cost) += l->object_scale * pow(1 - l->output[p_index], 2);
        avg_obj += l->output[p_index];
        l->delta[p_index] = l->object_scale * (1. - l->output[p_index]);

        if (l->rescore)
        {
          l->delta[p_index] = l->object_scale * (iou - l->output[p_index]);
        }

        l->delta[box_index + 0] =
            l->coord_scale *
            (state.truth[tbox_index + 0] - l->output[box_index + 0]);
        l->delta[box_index + 1] =
            l->coord_scale *
            (state.truth[tbox_index + 1] - l->output[box_index + 1]);
        l->delta[box_index + 2] =
            l->coord_scale *
            (state.truth[tbox_index + 2] - l->output[box_index + 2]);
        l->delta[box_index + 3] =
            l->coord_scale *
            (state.truth[tbox_index + 3] - l->output[box_index + 3]);
        if (l->sqrt)
        {
          l->delta[box_index + 2] =
              l->coord_scale *
              (sqrt(state.truth[tbox_index + 2]) - l->output[box_index + 2]);
          l->delta[box_index + 3] =
              l->coord_scale *
              (sqrt(state.truth[tbox_index + 3]) - l->output[box_index + 3]);
        }

        *(l->cost) += pow(1 - iou, 2);
        avg_iou += iou;
        ++count;
      }
    }

    if (0)
    {
      float* costs =
          (float*)xcalloc(l->batch * locations * l->n, sizeof(float));
      for (b = 0; b < l->batch; ++b)
      {
        int index = b * l->inputs;
        for (i = 0; i < locations; ++i)
        {
          for (j = 0; j < l->n; ++j)
          {
            int p_index = index + locations * l->classes + i * l->n + j;
            costs[b * locations * l->n + i * l->n + j] =
                l->delta[p_index] * l->delta[p_index];
          }
        }
      }
      int indexes[100];
      top_k(costs, l->batch * locations * l->n, 100, indexes);
      float cutoff = costs[indexes[99]];
      for (b = 0; b < l->batch; ++b)
      {
        int index = b * l->inputs;
        for (i = 0; i < locations; ++i)
        {
          for (j = 0; j < l->n; ++j)
          {
            int p_index = index + locations * l->classes + i * l->n + j;
            if (l->delta[p_index] * l->delta[p_index] < cutoff)
              l->delta[p_index] = 0;
          }
        }
      }
      free(costs);
    }

    *(l->cost) = pow(mag_array(l->delta, l->outputs * l->batch), 2);

    printf(
        "Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any "
        "Obj: %f, count: %d\n",
        avg_iou / count, avg_cat / count, avg_allcat / (count * l->classes),
        avg_obj / count, avg_anyobj / (l->batch * locations * l->n), count);
  }
}

void BackwardDetectionLayer(layer* l, NetworkState state)
{
  axpy_cpu(l->batch * l->inputs, 1, l->delta, 1, state.delta, 1);
}

void GetDetectionDetections(
    layer const* l, int w, int h, float thresh, Detection* dets)
{
  float const* predictions = l->output;
  for (int i = 0; i < l->side * l->side; ++i)
  {
    int row = i / l->side;
    int col = i % l->side;
    for (int n = 0; n < l->n; ++n)
    {
      int index = i * l->n + n;
      int p_index = l->side * l->side * l->classes + i * l->n + n;
      float scale = predictions[p_index];
      int box_index =
          l->side * l->side * (l->classes + l->n) + (i * l->n + n) * 4;
      Box b;
      b.x = (predictions[box_index + 0] + col) / l->side * w;
      b.y = (predictions[box_index + 1] + row) / l->side * h;
      b.w = pow(predictions[box_index + 2], (l->sqrt ? 2 : 1)) * w;
      b.h = pow(predictions[box_index + 3], (l->sqrt ? 2 : 1)) * h;
      dets[index].bbox = b;
      dets[index].objectness = scale;
      for (int j = 0; j < l->classes; ++j)
      {
        int class_index = i * l->classes;
        float prob = scale * predictions[class_index + j];
        dets[index].prob[j] = (prob > thresh) ? prob : 0;
      }
    }
  }
}

#ifdef GPU
void ForwardDetectionLayerGpu(layer* l, NetworkState state)
{
  if (!state.train)
  {
    copy_ongpu(l->batch * l->inputs, state.input, 1, l->output_gpu, 1);
    return;
  }

  float* in_cpu = (float*)xcalloc(l->batch * l->inputs, sizeof(float));
  float* truth_cpu = 0;
  if (state.truth)
  {
    int num_truth = l->batch * l->side * l->side * (1 + l->coords + l->classes);
    truth_cpu = (float*)xcalloc(num_truth, sizeof(float));
    cuda_pull_array(state.truth, truth_cpu, num_truth);
  }
  cuda_pull_array(state.input, in_cpu, l->batch * l->inputs);
  NetworkState cpu_state = state;
  cpu_state.train = state.train;
  cpu_state.truth = truth_cpu;
  cpu_state.input = in_cpu;
  ForwardDetectionLayer(l, cpu_state);
  cuda_push_array(l->output_gpu, l->output, l->batch * l->outputs);
  cuda_push_array(l->delta_gpu, l->delta, l->batch * l->inputs);
  free(cpu_state.input);
  if (cpu_state.truth)
    free(cpu_state.truth);
}

void BackwardDetectionLayerGpu(layer* l, NetworkState state)
{
  axpy_ongpu(l->batch * l->inputs, 1, l->delta_gpu, 1, state.delta, 1);
}
#endif