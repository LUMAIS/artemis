//#include <cuda_provider_factory.h>
//#include <onnxruntime_cxx_api.h>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <torch/torch.h>
#include <torch/script.h> 

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

struct Detection {
    cv::Rect bbox;
    float score;
    int class_idx;
};

uint16_t extr = 205;        // sidebar size
uint16_t half_imgsize = 80; // area half size for a moving object
const uint16_t model_resolution = 992;  // frame resizing for model (992)
uint16_t frame_resolution = 992;//frame frame_resolution
uint16_t reduseres = 290;   // (good value 248)
uint16_t color_threshold = 70; // 65-70


std::string class_name[9] = {"ta", "a", "ah", "tl", "l", "fn", "u", "p", "b"};

class IMGsamples
{
public:
  uint16_t num;
  std::vector<cv::Mat> samples;
  std::vector<cv::Point2f> coords; // coordinates relative to the center of the cluster

  IMGsamples(std::vector<cv::Mat> samples, std::vector<cv::Point2f> coords)
  {
    this->samples = samples;
    this->coords = coords;
    num = samples.size();
  }
};

class OBJdetect
{
public:
  cv::Point2f detect;
  cv::Point2f rectangle;
  std::string type;
};

class ClsObjR
{
  public:
  size_t cls_id;
  size_t obj_id;
  float r;
};

class ALObject // AntLab Object
{
public:
  uint16_t id;
  std::string obj_type;
  std::vector<cv::Point2f> claster_points;
  cv::Point2f claster_center;
  cv::Point2f model_center;
  bool det_mc = false;

  std::vector<cv::Point2f> track_points;
  cv::Point2f rectangle;
  cv::Mat img;

  std::vector<cv::Mat> samples;
  std::vector<cv::Point2f> coords;
  std::vector<IMGsamples> moution_samples;
  bool det_pos = false;

  ALObject(uint16_t id, std::string obj_type, std::vector<cv::Point2f> claster_points, cv::Mat img)
  {
    this->obj_type = obj_type;
    this->id = id;
    this->claster_points = claster_points;
    this->img = img;
    center_determine(true);
    //model_center = claster_center;
  }

  void center_determine(bool samplescreation)
  {

    std::vector<cv::Point2f> cp;
    std::vector<float> l;

    for(int i=0; i < claster_points.size(); i++)
    {
      for(int j=i; j< claster_points.size(); j++)
      {
        cv::Point2f p;
        float r;

        p.x = (claster_points.at(i).x + claster_points.at(j).x)/2;
        p.y = (claster_points.at(i).y + claster_points.at(j).y)/2;

        cp.push_back(p);

        r = sqrt(pow((claster_points.at(i).x - claster_points.at(j).x),2) + pow((claster_points.at(i).y - claster_points.at(j).y),2));

        l.push_back(r);
      }
    }

    cv::Point2f sumcp;
    float suml = 0;

    sumcp.x = 0;
    sumcp.y = 0;

    for(int i =0; i< cp.size(); i++)
    {
      sumcp.x += cp.at(i).x*l.at(i);
      sumcp.y += cp.at(i).y*l.at(i);
      suml += l.at(i);
    }

    claster_center.x = sumcp.x/suml;
    claster_center.y = sumcp.y/suml;

    det_pos = true;


    if (samplescreation == true)
    {
      samples_creation();
    }
  }

  void samples_creation()
  {
    cv::Mat sample;
    cv::Point2f coord;

    samples.clear();
    coords.clear();

    for (int i = 0; i < claster_points.size(); i++)
    {
      coord.y = claster_points.at(i).y - claster_center.y + half_imgsize;
      coord.x = claster_points.at(i).x - claster_center.x + half_imgsize;

      if (coord.y > 0 && (coord.y + frame_resolution / reduseres) < 2 * half_imgsize && coord.x > 0 && (coord.x + frame_resolution / reduseres) < 2 * half_imgsize)
      {
        sample = img(cv::Range(coord.y, coord.y + frame_resolution / reduseres), cv::Range(coord.x, coord.x + frame_resolution / reduseres));
        samples.push_back(sample);
        coords.push_back(coord);
      }
    }

    IMGsamples buf(samples, coords);
    moution_samples.push_back(buf);
  }

  void push_track_point(cv::Point2f track_point)
  {
    track_points.push_back(track_point);

    if (track_points.size() > 33)
      track_points.erase(track_points.begin());
  }

  cv::Point2f proposed_center()
  {
    cv::Point2f proposed;

    if (track_points.size() > 1)
    {
      proposed.x = claster_center.x + 0.5*(claster_center.x - track_points.at(track_points.size() - 2).x);
      proposed.y = claster_center.y + 0.5*(claster_center.y - track_points.at(track_points.size() - 2).y);
    }
    else
      proposed = claster_center;

    return proposed;
  }
};

cv::Mat frame_resizing(cv::Mat frame);
std::vector<std::pair<cv::Point2f,uint16_t>> DetectorMotionV2_1(std::string pathmodel, torch::DeviceType device_type, cv::Mat frame0, cv::Mat frame, std::vector<ALObject> &objects, size_t id_frame, bool usedetector);