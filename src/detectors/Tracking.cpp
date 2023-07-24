#include "Tracking.hpp"

#include <unistd.h>

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

//------------------------May 4 2022-------------------------

cv::Mat frame_resizing(cv::Mat frame)
{
  uint16_t rows = frame.rows;
  uint16_t cols = frame.cols;

  float rwsize;
  float clsize;

  if (rows > cols)
  {
    rwsize = (float)model_resolution * rows / cols;
    clsize = (float)model_resolution;
  }
  else
  {
    rwsize = (float)model_resolution;
    clsize = (float)model_resolution * cols / rows;
  }

  cv::resize(frame, frame, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);
  cv::Rect rect(0, 0, model_resolution, model_resolution);

  return frame(rect);
}

std::vector<OBJdetect> detectorV4(std::string pathmodel, cv::Mat frame, torch::DeviceType device_type)
{
  std::vector<OBJdetect> obj_detects;
  auto millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  torch::jit::script::Module module = torch::jit::load(pathmodel);
  std::cout << "Load module +" << duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - millisec << "ms" << std::endl;
  millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

  uint16_t pointsdelta = 5;
  std::vector<cv::Point2f> detects;
  std::vector<cv::Point2f> detectsCent;
  std::vector<cv::Point2f> detectsRect;
  std::vector<uint8_t> Objtype;

  cv::Scalar class_name_color[9] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255), cv::Scalar(255, 255, 0), cv::Scalar(255, 255, 255), cv::Scalar(200, 0, 200), cv::Scalar(100, 0, 255)};
  cv::Mat imageBGR;

  frame.copyTo(imageBGR);

  if(imageBGR.channels() == 1) {
    cv::cvtColor(imageBGR, imageBGR, cv::COLOR_GRAY2RGB);
  } else {
    assert(imageBGR.type() == CV_8UC3 && "Color input frames in the BGR (OpenCV-native) format is expected");
    cv::cvtColor(imageBGR, imageBGR, cv::COLOR_BGR2RGB);
  }
  imageBGR.convertTo(imageBGR, CV_32FC3, 1.0f / 255.0f);
  auto input_tensor = torch::from_blob(imageBGR.data, {1, imageBGR.rows, imageBGR.cols, 3});
  input_tensor = input_tensor.permute({0, 3, 1, 2}).contiguous();
  input_tensor = input_tensor.to(device_type);
  //----------------------------------
  // module.to(device_type);

  if (device_type != torch::kCPU)
  {
    input_tensor = input_tensor.to(torch::kHalf);
  }
  //----------------------------------

  // std::cout<<"input_tensor.to(device_type) - OK"<<std::endl;
  std::vector<torch::jit::IValue> input;
  input.emplace_back(input_tensor);
  // std::cout<<"input.emplace_back(input_tensor) - OK"<<std::endl;

  millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  auto outputs = module.forward(input).toTuple();
  // std::cout << "Processing +" << duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - millisec << "ms" << std::endl;
  millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

  // std::cout<<"module.forward(input).toTuple() - OK"<<std::endl;
  torch::Tensor detections = outputs->elements()[0].toTensor();

  int item_attr_size = 13;
  int batch_size = detections.size(0);
  auto num_classes = detections.size(2); // - item_attr_size;

  auto conf_thres = 0.50;
  auto conf_mask = detections.select(2, 4).ge(conf_thres).unsqueeze(2);

  std::vector<std::vector<Detection>> output;
  output.reserve(batch_size);

  for (int batch_i = 0; batch_i < batch_size; batch_i++)
  {
    // apply constrains to get filtered detections for current image
    auto det = torch::masked_select(detections[batch_i], conf_mask[batch_i]).view({-1, num_classes});
    // if none detections remain then skip and start to process next image

    if (0 == det.size(0))
    {
      continue;
    }

    for (size_t i = 0; i < det.size(0); ++i)
    {
      float x = det[i][0].item().toFloat() * imageBGR.cols / model_resolution;
      float y = det[i][1].item().toFloat() * imageBGR.rows / model_resolution;

      float h = det[i][2].item().toFloat() * imageBGR.cols / model_resolution;
      float w = det[i][3].item().toFloat() * imageBGR.rows / model_resolution;

      float wheit = 0;
      Objtype.push_back(8);

      for (int j = 4; j < det.size(1); j++)
      {
        if (det[i][j].item().toFloat() > wheit)
        {
          wheit = det[i][j].item().toFloat();
          Objtype.at(i) = j - 4;
        }
      }

      detectsCent.push_back(cv::Point(x, y));
      detectsRect.push_back(cv::Point(h, w));
    }
  }

  for (size_t i = 0; i < detectsCent.size(); i++)
  {
    if (detectsCent.at(i).x > 0)
    {
      for (size_t j = 0; j < detectsCent.size(); j++)
      {
        if (detectsCent.at(j).x > 0 && i != j)
        {
          if (sqrt(pow(detectsCent.at(i).x - detectsCent.at(j).x, 2) + pow(detectsCent.at(i).y - detectsCent.at(j).y, 2)) < pointsdelta)
          {
            detectsCent.at(i).x = (detectsCent.at(i).x + detectsCent.at(j).x) * 1.0 / 2;
            detectsCent.at(i).y = (detectsCent.at(i).y + detectsCent.at(j).y) * 1.0 / 2;

            detectsRect.at(i).x = (detectsRect.at(i).x + detectsRect.at(j).x) * 1.0 / 2;
            detectsRect.at(i).y = (detectsRect.at(i).y + detectsRect.at(j).y) * 1.0 / 2;

            detectsCent.at(j).x = -1;
          }
        }
      }
    }
  }

  for (size_t i = 0; i < detectsCent.size(); i++)
  {

    cv::Point2f pt1;
    cv::Point2f pt2;
    cv::Point2f ptext;

    if (detectsCent.at(i).x >= 0)
    {

      OBJdetect obj_buf;

      obj_buf.detect = detectsCent.at(i);
      obj_buf.rectangle = detectsRect.at(i);
      obj_buf.type = class_name[Objtype.at(i)];
      obj_detects.push_back(obj_buf);

      detects.push_back(detectsCent.at(i));
      pt1.x = detectsCent.at(i).x - detectsRect.at(i).x / 2;
      pt1.y = detectsCent.at(i).y - detectsRect.at(i).y / 2;

      pt2.x = detectsCent.at(i).x + detectsRect.at(i).x / 2;
      pt2.y = detectsCent.at(i).y + detectsRect.at(i).y / 2;

      ptext.x = detectsCent.at(i).x - 5;
      ptext.y = detectsCent.at(i).y + 5;

      rectangle(imageBGR, pt1, pt2, class_name_color[Objtype.at(i)], 1);

      cv::putText(imageBGR,                  // target image
                  class_name[Objtype.at(i)], // text
                  ptext,                     // top-left position
                  1,
                  0.8,
                  class_name_color[Objtype.at(i)], // font color
                  1);
    }
  }

  millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  return obj_detects;
}

cv::Mat color_correction(cv::Mat imag)
{
  cv::Mat imagchange;

  imag.copyTo(imagchange);

  for (int y = 0; y < imag.rows; y++)
  {
    for (int x = 0; x < imag.cols; x++)
    {
      uchar color1 = imag.at<uchar>(cv::Point(x, y));

      if (color1 < (uchar)color_threshold) 
        imagchange.at<uchar>(cv::Point(x, y)) = 0;
      else
        imagchange.at<uchar>(cv::Point(x, y)) = 255;
    }
  }

  // imshow("imagchange", imagchange);
  // cv::waitKey(0);

  return imagchange;
}

bool compare_clsobj(ClsObjR a, ClsObjR b)
{
  if (a.r < b.r)
    return 1;
  else
    return 0;
}

cv::Point2f claster_center(std::vector<cv::Point2f> claster_points)
{
  cv::Point2f claster_center;

  int powx = 0;
  int powy = 0;

  for (int i = 0; i < claster_points.size(); i++)
  {
    powx = powx + pow(claster_points[i].x, 2);
    powy = powy + pow(claster_points[i].y, 2);
  }

  claster_center.x = sqrt(powx / claster_points.size());
  claster_center.y = sqrt(powy / claster_points.size());

  return claster_center;
}

std::vector<std::pair<cv::Point2f,uint16_t>> DetectorMotionV2_1(std::string pathmodel, torch::DeviceType device_type, cv::Mat frame0, cv::Mat frame, std::vector<ALObject> &objects, size_t id_frame, bool usedetector)
{
  cv::Scalar class_name_color[20] = {
      cv::Scalar(255, 0, 0),
      cv::Scalar(0, 20, 200),
      cv::Scalar(0, 255, 0),
      cv::Scalar(255, 0, 255),
      cv::Scalar(0, 255, 255),
      cv::Scalar(255, 255, 0),
      cv::Scalar(255, 255, 255),
      cv::Scalar(200, 0, 200),
      cv::Scalar(100, 0, 255),
      cv::Scalar(255, 0, 100),
      cv::Scalar(30, 20, 200),
      cv::Scalar(25, 255, 0),
      cv::Scalar(255, 44, 255),
      cv::Scalar(88, 255, 255),
      cv::Scalar(255, 255, 39),
      cv::Scalar(255, 255, 255),
      cv::Scalar(200, 46, 200),
      cv::Scalar(100, 79, 255),
      cv::Scalar(200, 46, 150),
      cv::Scalar(140, 70, 205),
  };

  std::vector<std::vector<cv::Point2f>> clasters;
  std::vector<cv::Point2f> motion;
  std::vector<cv::Mat> imgs;

  cv::Mat imageBGR0;
  cv::Mat imageBGR;

  cv::Mat imag;
  cv::Mat imagbuf;

  float corr = 1.0;
  if (usedetector)
  {
    corr = (float)frame.rows / (float)model_resolution;
    frame = frame_resizing(frame);
  }
  
  cv::Mat framebuf = frame;

  uint16_t rows = frame.rows;
  uint16_t cols = frame.cols;

  frame_resolution = rows;

  float koef = (float)rows / (float)model_resolution;

  uint8_t mpct = 3 * koef;      // minimum number of points for a cluster (tracking object) (good value 5)
  uint8_t mpcc = 7 * koef;      // minimum number of points for a cluster (creation new object) (good value 13)
  float nd = 1.5 * koef;    //(good value 6-15)
  uint8_t rcobj = 15 * koef;    //(good value 15)
  float robj = 22.0 * koef; //(good value 17)
  float robj_k = 1.0;
  uint8_t mdist = 10 * koef; // maximum distance from cluster center (good value 10)
  uint8_t pft = 1;    // points fixation threshold (good value 9)

  cv::Mat img;

  std::vector<OBJdetect> detects;

  //--------------------<detection using a classifier>----------
  if (usedetector)
  {
    detects = detectorV4(pathmodel, frame, device_type);

    for (uint16_t i = 0; i < objects.size(); i++)
    {
      objects[i].det_mc = false;
    }

    for (uint16_t i = 0; i < detects.size(); i++)
    {
      if (detects.at(i).type != "a")
      {
        detects.erase(detects.begin() + i);
        i--;
      }
    }

    for (uint16_t i = 0; i < detects.size(); i++)
    {
      std::vector<cv::Point2f> claster_points;
      claster_points.push_back(detects.at(i).detect);
      img = framebuf(cv::Range(detects.at(i).detect.y - half_imgsize * koef, detects.at(i).detect.y + half_imgsize * koef), cv::Range(detects.at(i).detect.x - half_imgsize * koef, detects.at(i).detect.x + half_imgsize * koef));
      
      // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
      // img.convertTo(img, CV_8UC3);

      ALObject obj(objects.size(), detects.at(i).type, claster_points, img);
      obj.model_center = detects.at(i).detect;
      obj.claster_center = detects.at(i).detect;
      obj.rectangle = detects.at(i).rectangle;
      obj.det_mc = true;

      float rm = rcobj * (float)rows / (float)reduseres;
      bool newobj = true;
      uint16_t n;

      if (objects.size() > 0)
      {
        rm = sqrt(pow((objects[0].claster_center.x - obj.claster_center.x), 2) + pow((objects[0].claster_center.y - obj.claster_center.y), 2));
        // rm = sqrt(pow((objects[0].proposed_center().x - obj.claster_center.x), 2) + pow((objects[0].proposed_center().y - obj.claster_center.y), 2));
        if (rm < rcobj * (float)rows / (float)reduseres)
        {
          n = 0;
          newobj = false;
        }
      }

      for (uint16_t j = 1; j < objects.size(); j++)
      {
        float r = sqrt(pow((objects[j].claster_center.x - obj.claster_center.x), 2) + pow((objects[j].claster_center.y - obj.claster_center.y), 2));
        // float r = sqrt(pow((objects[j].proposed_center().x - obj.claster_center.x), 2) + pow((objects[j].proposed_center().y - obj.claster_center.y), 2));
        if (r < rcobj * (float)rows/ (float)reduseres && r < rm)
        {
          rm = r;
          n = j;
          newobj = false;
        }
      }

      if (newobj == false)
      {
        objects[n].model_center = obj.model_center;
        objects[n].rectangle = obj.rectangle;
        objects[n].img = obj.img;
        objects[n].det_mc = true;
      }
      else
      {

        for (size_t j = 0; j < objects.size(); j++)
        {
          if(objects[i].det_pos == false)
           continue;
           
          float r = sqrt(pow((objects[j].claster_center.x - obj.claster_center.x), 2) + pow((objects[j].claster_center.y - obj.claster_center.y), 2));
          if (r < (float)robj * 2.3 * (float)rows / (float)reduseres)
          {
            newobj = false;
            break;
          }
        }

        if(newobj == true)
          objects.push_back(obj);
      }
    }
  }
  //--------------------</detection using a classifier>---------

  //--------------------<moution detections>--------------------

  float rwsize;
  float clsize;

  frame.copyTo(imagbuf);

  if (rows > cols)
  {
    rwsize = (float)frame_resolution * rows / (float)cols;
    clsize = (float)rows;
  }
  else
  {
    rwsize = (float)rows;
    clsize = (float)frame_resolution * cols / (float)rows;
  }
  
  cv::resize(imagbuf, imagbuf, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);
  cv::Rect rectb(0, 0, rows, rows);
  imag = imagbuf(rectb);

  if(imag.channels() > 1)
    cv::cvtColor(imag, imag, cv::COLOR_BGR2GRAY);
  // if(imag.channels() == 1)
  //   cv::cvtColor(imag, imag, cv::COLOR_GRAY2RGB);
  // else cv::cvtColor(imag, imag, cv::COLOR_BGR2RGB);
  // imag.convertTo(imag, CV_8UC3);

  if (rows > cols)
  {
    rwsize = (float)reduseres * rows / (float)cols;
    clsize = reduseres;
  }
  else
  {
    rwsize = reduseres;
    clsize = (float)reduseres * cols / (float)rows;
  }

  cv::resize(frame0, frame0, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);
  cv::Rect rect0(0, 0, reduseres, reduseres);
  imageBGR0 = frame0(rect0);
  if(imageBGR0.channels() > 1)
     cv::cvtColor(imageBGR0, imageBGR0, cv::COLOR_BGR2GRAY);
  // imageBGR0.convertTo(imageBGR0, CV_8UC1);

  cv::resize(frame, frame, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);

  cv::Rect rect(0, 0, reduseres, reduseres);
  imageBGR = frame(rect);

  if(imageBGR0.channels() > 1)
     cv::cvtColor(imageBGR0, imageBGR0, cv::COLOR_BGR2GRAY);
  // imageBGR.convertTo(imageBGR, CV_8UC1);

  cv::Point2f pm;

  imageBGR0 = color_correction(imageBGR0);
  imageBGR = color_correction(imageBGR);

  for (uint16_t y = 0; y < imageBGR0.rows; y++)
  {
    for (uint16_t x = 0; x < imageBGR0.cols; x++)
    {
      uchar color1 = imageBGR0.at<uchar>(cv::Point(x, y));
      uchar color2 = imageBGR.at<uchar>(cv::Point(x, y));

      if (((int)color2 - (int)color1) > pft)
      {
        pm.x = (float)x * rows / reduseres;
        pm.y = (float)y * rows / reduseres;
        motion.push_back(pm);
      }
    }
  }

  cv::Point2f pt1;
  cv::Point2f pt2;

  uint16_t ncls = 0;
  uint16_t nobj;

  if (objects.size() > 0)
    nobj = 0;
  else
    nobj = -1;
  //--------------</moution detections>--------------------

  //--------------<claster creation>-----------------------

  while (motion.size() > 0)
  {
    cv::Point2f pc;

    if (nobj > -1 && nobj < objects.size())
    {
      pc = objects[nobj].claster_center;
      // pc = objects[nobj].proposed_center();
      nobj++;
    }
    else
    {
      pc = motion.at(0);
      motion.erase(motion.begin());
    }

    clasters.push_back(std::vector<cv::Point2f>());
    clasters[ncls].push_back(pc);

    for (int i = 0; i < motion.size(); i++)
    {
      float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));
      if (r < nd * (float)rows / (float)reduseres)
      {
        cv::Point2f cl_c = claster_center(clasters.at(ncls));
        r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
        if (r < (float)mdist * (float)rows / (float)reduseres)
        {
          clasters.at(ncls).push_back(motion.at(i));
          motion.erase(motion.begin() + i);
          i--;
        }
      }
    }

    uint16_t newp;
    do
    {
      newp = 0;

      for (uint16_t c = 0; c < clasters[ncls].size(); c++)
      {
        pc = clasters[ncls].at(c);
        for (int i = 0; i < motion.size(); i++)
        {
          float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));

          if (r < nd * (float)rows / (float)reduseres)
          {
            cv::Point2f cl_c = claster_center(clasters.at(ncls));
            r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
            if (r < (float)mdist * (float)rows / (float)reduseres)
            {
              clasters.at(ncls).push_back(motion.at(i));
              motion.erase(motion.begin() + i);
              i--;
              newp++;
            }
          }
        }
      }
    } while (newp > 0 && motion.size() > 0);

    ncls++;
  }
  //--------------</claster creation>----------------------

  //--------------<clusters to objects>--------------------
  if (objects.size() > 0)
  {
    for (size_t i = 0; i < objects.size(); i++)
      objects[i].det_pos = false;

    std::vector<ClsObjR> clsobjrs;
    ClsObjR clsobjr;
    for (size_t i = 0; i < objects.size(); i++)
    {
      for (size_t cls = 0; cls < clasters.size(); cls++)
      {
        clsobjr.cls_id = cls;
        clsobjr.obj_id = i;
        cv::Point2f clastercenter = claster_center(clasters[cls]);
        // clsobjr.r = sqrt(pow((objects[i].claster_center.x - clastercenter.x), 2) + pow((objects[i].claster_center.y - clastercenter.y), 2));
        clsobjr.r = sqrt(pow((objects[i].proposed_center().x - clastercenter.x), 2) + pow((objects[i].proposed_center().y - clastercenter.y), 2));
        clsobjrs.push_back(clsobjr);
      }
    }

    sort(clsobjrs.begin(), clsobjrs.end(), compare_clsobj);

    //--<corr obj use model>---
    if (usedetector == true)
      for (size_t i = 0; i < clsobjrs.size(); i++)
      {
        size_t cls_id = clsobjrs.at(i).cls_id;
        size_t obj_id = clsobjrs.at(i).obj_id;

        if (objects.at(obj_id).det_mc == true)
        {
          cv::Point2f clastercenter = claster_center(clasters[cls_id]);
          pt1.x = objects.at(obj_id).model_center.x - objects.at(obj_id).rectangle.x / 2;
          pt1.y = objects.at(obj_id).model_center.y - objects.at(obj_id).rectangle.y / 2;

          pt2.x = objects.at(obj_id).model_center.x + objects.at(obj_id).rectangle.x / 2;
          pt2.y = objects.at(obj_id).model_center.y + objects.at(obj_id).rectangle.y / 2;

          if (pt1.x < clastercenter.x && clastercenter.x < pt2.x && pt1.y < clastercenter.y && clastercenter.y < pt2.y)
          {

            if (objects[obj_id].det_pos == false)
              objects[obj_id].claster_points = clasters.at(cls_id);
            else
            {
              for (size_t j = 0; j < clasters.at(cls_id).size(); j++)
                objects[obj_id].claster_points.push_back(clasters.at(cls_id).at(j));
            }

            objects[obj_id].center_determine(false);

            for (size_t j = 0; j < clsobjrs.size(); j++)
            {
              if (clsobjrs.at(j).cls_id == cls_id)
              {
                clsobjrs.erase(clsobjrs.begin() + j);
                j--;
              }
            }
            i = 0;
          }
        }
      }
    //--</corr obj use model>---

    //---<det obj>---
    for (size_t i = 0; i < clsobjrs.size(); i++)
    {
      size_t cls_id = clsobjrs.at(i).cls_id;
      size_t obj_id = clsobjrs.at(i).obj_id;

      if (clsobjrs.at(i).r < (float)robj * (float)rows/ (float)reduseres && clasters.at(cls_id).size() > mpct)
      {

        if (objects[obj_id].det_pos == false)
          objects[obj_id].claster_points = clasters.at(cls_id);
        else
        {
          continue;
          // for (size_t j = 0; j < clasters.at(cls_id).size(); j++)
          //   objects[obj_id].claster_points.push_back(clasters.at(cls_id).at(j));
        }

        objects[obj_id].center_determine(false);

        for (size_t j = 0; j < clsobjrs.size(); j++)
        {
          if (clsobjrs.at(j).cls_id == cls_id)
          {
            clsobjrs.erase(clsobjrs.begin() + j);
            j--;
          }
        }
        i = 0;
      }
    }
    //---</det obj>---
    //---<new obj>----
    for (size_t i = 0; i < clsobjrs.size(); i++)
    {
      size_t cls_id = clsobjrs.at(i).cls_id;
      size_t obj_id = clsobjrs.at(i).obj_id;

      cv::Point2f clastercenter = claster_center(clasters[cls_id]);
      bool newobj = true;

      for (size_t j = 0; j < objects.size(); j++)
      {
        float r = sqrt(pow((objects[j].claster_center.x - clastercenter.x), 2) + pow((objects[j].claster_center.y - clastercenter.y), 2));
        if (r < (float)robj * 2.3 * (float)rows/ (float)reduseres)
        {
          newobj = false;
          break;
        }
      }

      if (clasters[cls_id].size() > mpcc && newobj == true) // if there are enough moving points
      {
        // framebuf.convertTo(imagbuf, CV_8UC3);
        framebuf.copyTo(imagbuf);

        img = imagbuf(cv::Range(clastercenter.y - half_imgsize * koef, clastercenter.y + half_imgsize * koef), cv::Range(clastercenter.x - half_imgsize * koef, clastercenter.x + half_imgsize * koef));
        // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        // img.convertTo(img, CV_8UC3);
        ALObject obj(objects.size(), "a", clasters[cls_id], img);
        objects.push_back(obj);
        for (size_t j = 0; j < clsobjrs.size(); j++)
        {
          if (clsobjrs.at(j).cls_id == cls_id)
          {
            clsobjrs.erase(clsobjrs.begin() + j);
            j--;
          }
        }
        i = 0;
      }
    }
    //--</new obj>--

    //--<corr obj>---
    for (size_t i = 0; i < clsobjrs.size(); i++)
    {
      size_t cls_id = clsobjrs.at(i).cls_id;
      size_t obj_id = clsobjrs.at(i).obj_id;

      if (clsobjrs.at(i).r < (float)robj * robj_k * (float)rows/ (float)reduseres && clasters.at(cls_id).size() > mpct / 2)
      {
        for (size_t j = 0; j < clasters.at(cls_id).size(); j++)
          objects[obj_id].claster_points.push_back(clasters.at(cls_id).at(j));

        objects[obj_id].center_determine(false);

        for (size_t j = 0; j < clsobjrs.size(); j++)
        {
          if (clsobjrs.at(j).cls_id == cls_id)
          {
            clsobjrs.erase(clsobjrs.begin() + j);
            j--;
          }
        }
        i = 0;
      }
    }
    //--</corr obj>---
  }
  else
  {
    //--<new obj>--
    for (int cls = 0; cls < clasters.size(); cls++)
    {
      cv::Point2f clastercenter = claster_center(clasters[cls]);
      bool newobj = true;

      for (int i = 0; i < objects.size(); i++)
      {
        float r = sqrt(pow((objects[i].claster_center.x - clastercenter.x), 2) + pow((objects[i].claster_center.y - clastercenter.y), 2));
        if (r < (float)robj * (float)rows / (float)reduseres)
        {
          newobj = false;
          break;
        }
      }

      if (clasters[cls].size() > mpcc && newobj == true) // if there are enough moving points
      {
        // framebuf.convertTo(imagbuf, CV_8UC3);
        framebuf.copyTo(imagbuf);
        img = imagbuf(cv::Range(clastercenter.y - half_imgsize * koef, clastercenter.y + half_imgsize * koef), cv::Range(clastercenter.x - half_imgsize * koef, clastercenter.x + half_imgsize * koef));
        // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        // img.convertTo(img, CV_8UC3);
        ALObject obj(objects.size(), "a", clasters[cls], img);
        objects.push_back(obj);
        clasters.erase(clasters.begin() + cls);
        cls--;

        if (cls < 0)
          cls = 0;
      }
    }
    //--</new obj>--
  }
  //--------------</clusters to objects>-------------------

  //--------------<post processing>-----------------------
  for (int i = 0; i < objects.size(); i++)
  {
    if (objects.at(i).det_mc == false && objects.at(i).det_pos == false)
      continue;

    // framebuf.convertTo(imagbuf, CV_8UC3);
    framebuf.copyTo(imagbuf);

    if (objects[i].det_mc == false)
    {
      pt1.y = objects[i].claster_center.y - half_imgsize * koef;
      pt2.y = objects[i].claster_center.y + half_imgsize * koef;

      pt1.x = objects[i].claster_center.x - half_imgsize * koef;
      pt2.x = objects[i].claster_center.x + half_imgsize * koef;
    }
    else
    {
      pt1.y = objects[i].model_center.y - half_imgsize * koef;
      pt2.y = objects[i].model_center.y + half_imgsize * koef;

      pt1.x = objects[i].model_center.x - half_imgsize * koef;
      pt2.x = objects[i].model_center.x + half_imgsize * koef;
    }

    if (pt1.y < 0)
      pt1.y = 0;

    if (pt2.y > imagbuf.rows)
      pt2.y = imagbuf.rows;

    if (pt1.x < 0)
      pt1.x = 0;

    if (pt2.x > imagbuf.cols)
      pt2.x = imagbuf.cols;

    // std::cout << "<post processing 2>" << std::endl;
    // std::cout << "pt1 - " << pt1 << std::endl;
    // std::cout << "pt2 - " << pt2 << std::endl;

    img = imagbuf(cv::Range(pt1.y, pt2.y), cv::Range(pt1.x, pt2.x));
    // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    // img.convertTo(img, CV_8UC3);
    img.copyTo(objects[i].img);
    objects[i].center_determine(true);

    if (objects[i].det_mc == false)
      objects[i].push_track_point(objects[i].claster_center);
    else
      objects[i].push_track_point(objects[i].model_center);
  }
  
  /*/--------------<visualization>--------------------------
  for (int i = 0; i < objects.size(); i++)
  {
    for (int j = 0; j < objects.at(i).claster_points.size(); j++) // visualization of the claster_points
    {
      pt1.x = objects.at(i).claster_points.at(j).x;
      pt1.y = objects.at(i).claster_points.at(j).y;

      pt2.x = objects.at(i).claster_points.at(j).x + (float)rows / (float)reduseres;
      pt2.y = objects.at(i).claster_points.at(j).y + (float)rows / (float)reduseres;

      rectangle(imag, pt1, pt2, class_name_color[objects.at(i).id], 1);
    }

    if (objects.at(i).det_mc == true) // visualization of the classifier
    {
      pt1.x = objects.at(i).model_center.x - objects.at(i).rectangle.x / 2;
      pt1.y = objects.at(i).model_center.y - objects.at(i).rectangle.y / 2;

      pt2.x = objects.at(i).model_center.x + objects.at(i).rectangle.x / 2;
      pt2.y = objects.at(i).model_center.y + objects.at(i).rectangle.y / 2;

      rectangle(imag, pt1, pt2, class_name_color[objects.at(i).id], 1);
    }

    for (int j = 0; j < objects.at(i).track_points.size(); j++)
      cv::circle(imag, objects.at(i).track_points.at(j), 1, class_name_color[objects.at(i).id], 2);
  }
  //--------------</visualization>-------------------------

  //--------------<baseimag>-------------------------------
  cv::Mat baseimag(rows, rows + extr * koef, CV_8UC3, cv::Scalar(0, 0, 0));
  for (int i = 0; i < objects.size(); i++)
  {
    std::string text = objects.at(i).obj_type + " ID" + std::to_string(objects.at(i).id);

    cv::Point2f ptext;
    ptext.x = 20;
    ptext.y = (30 + objects.at(i).img.cols) * objects.at(i).id + 20;

    cv::putText(baseimag, // target image
                text,     // text
                ptext,    // top-left position
                1,
                1,
                class_name_color[objects.at(i).id], // font color
                1);

    pt1.x = ptext.x - 1;
    pt1.y = ptext.y - 1 + 10;

    pt2.x = ptext.x + objects.at(i).img.cols + 1;
    pt2.y = ptext.y + objects.at(i).img.rows + 1 + 10;

    if (pt2.y < baseimag.rows && pt2.x < baseimag.cols)
    {
      rectangle(baseimag, pt1, pt2, class_name_color[objects.at(i).id], 1);
      objects.at(i).img.copyTo(baseimag(cv::Rect(pt1.x + 1, pt1.y + 1, objects.at(i).img.cols, objects.at(i).img.rows)));
    }
  }
  imag.copyTo(baseimag(cv::Rect(extr * koef, 0, imag.cols, imag.rows)));

  cv::Point2f p_idframe;
  p_idframe.x = rows + (extr - 95) * koef;
  p_idframe.y = 50;
  cv::putText(baseimag, std::to_string(id_frame), p_idframe, 1, 3, cv::Scalar(255, 255, 255), 2);
  // cv::cvtColor(baseimag, baseimag, cv::COLOR_BGR2RGB);
  cv::resize(baseimag, baseimag, cv::Size(992 + extr, 992), cv::InterpolationFlags::INTER_CUBIC);
  imshow("Motion", baseimag);
  cv::waitKey(1);
  //--------------</baseimag>-------------------------------*/
  
  std::vector<std::pair<cv::Point2f,uint16_t>> detects_P2f_id;

  for (int i = 0; i < objects.size(); i++)
  {
    if(objects[i].det_mc == true)
    {
      pt1.x = objects[i].model_center.x * corr;
      pt1.y = objects[i].model_center.y * corr;
      detects_P2f_id.push_back(std::make_pair(pt1,objects[i].id));
    }
    else if(objects[i].det_pos == true)
    {
      pt1.x = objects[i].claster_center.x * corr;
      pt1.y = objects[i].claster_center.y * corr;
      detects_P2f_id.push_back(std::make_pair(pt1,objects[i].id));
    }
      
  }
  
  return detects_P2f_id;
}




