#include "Trophallaxis.hpp"

#include <unistd.h>

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

std::vector<cv::Point2f> detectorT (torch::jit::script::Module module, cv::Mat frame, torch::DeviceType device_type)
{
  int resolution = 992;
  int pointsdelta = 30;
  std::vector<cv::Point2f> detects;
  std::vector<cv::Point2f> detectsbuf;
  cv::Mat imageBGR;
  //cv::resize(frame, imageBGR,cv::Size(992, 992),cv::InterpolationFlags::INTER_CUBIC);

  frame.copyTo(imageBGR);

  cv::cvtColor(imageBGR, imageBGR, cv::COLOR_BGR2RGB);
  imageBGR.convertTo(imageBGR, CV_32FC3, 1.0f / 255.0f);
  auto input_tensor = torch::from_blob(imageBGR.data, {1, imageBGR.rows, imageBGR.cols, 3});
  input_tensor = input_tensor.permute({0, 3, 1, 2}).contiguous();
  input_tensor = input_tensor.to(device_type);
  //----------------------------------
  //module.to(device_type);

  if (device_type != torch::kCPU) {
      //module.to(torch::kHalf);
      input_tensor = input_tensor.to(torch::kHalf);
      std::cout<<"....to(torch::kHalf)!!!"<<std::endl;
  }
  //----------------------------------

  std::cout<<"input_tensor.to(device_type) - OK"<<std::endl;
  std::vector<torch::jit::IValue> input;
  input.emplace_back(input_tensor);
  std::cout<<"input.emplace_back(input_tensor) - OK"<<std::endl;

  auto outputs = module.forward(input).toTuple();
  std::cout<<"module.forward(input).toTuple() - OK"<<std::endl;
  torch::Tensor detections = outputs->elements()[0].toTensor();

  int item_attr_size = 13;
  int batch_size = detections.size(0);
  auto num_classes = detections.size(2);// - item_attr_size;

  auto conf_thres = 0.60 ;
  auto conf_mask = detections.select(2, 4).ge(conf_thres).unsqueeze(2);

  std::vector<std::vector<Detection>> output;
  output.reserve(batch_size);

  for (int batch_i = 0; batch_i < batch_size; batch_i++) {
        // apply constrains to get filtered detections for current image
        auto det = torch::masked_select(detections[batch_i], conf_mask[batch_i]).view({-1, num_classes});

        // if none detections remain then skip and start to process next image

        //std::cout << "det.size(0) - " << det.size(0) << '\n';
        if (0 == det.size(0)) {
            continue;
        }

        for (size_t i=0; i < det.size(0); ++ i)
        {
            float left = det[i][0].item().toFloat() * imageBGR.cols / resolution;
            float top = det[i][1].item().toFloat() * imageBGR.rows / resolution;
            //float right = det[i][2].item().toFloat() * imageBGR.cols / resolution;
            //float bottom = det[i][3].item().toFloat() * imageBGR.rows / resolution;
            detectsbuf.push_back(cv::Point(left,top));
        }
  }
  for(size_t i=0; i < detectsbuf.size(); i++)
  {
    if(detectsbuf.at(i).x > 0) 
    {
      for(size_t j=0; j < detectsbuf.size(); j++)
      {
        if(detectsbuf.at(j).x > 0 && i != j)
        {
          if(sqrt(pow(detectsbuf.at(i).x - detectsbuf.at(j).x,2) + pow(detectsbuf.at(i).y - detectsbuf.at(j).y,2)) < pointsdelta)
          {
            detectsbuf.at(i).x = (detectsbuf.at(i).x + detectsbuf.at(j).x)*1.0/2;
            detectsbuf.at(i).y = (detectsbuf.at(i).y + detectsbuf.at(j).y)*1.0/2;
            detectsbuf.at(j).x = -1;
          }
        }
      }
    }
  }

  for(size_t i=0; i < detectsbuf.size(); i++)
  {
    if(detectsbuf.at(i).x >=0)
    {
      detects.push_back(detectsbuf.at(i));
    }
  }
  
  return detects;
}
