#include "Trophallaxis.hpp"

#include <unistd.h>

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

std::vector<cv::Point2f> detectorT (torch::jit::script::Module module, cv::Mat frame, torch::DeviceType device_type)
{
  const int  szml = 992;  // ML model resolution; TODO: make it a UI argument or fetch automatically from the ML model
  int  pointsdelta = 30;
  std::vector<cv::Point2f> detects;
  std::vector<cv::Point2f> detectsbuf;
  cv::Mat img;
  
  // Note: input frame is either gray-scale or color in BGR (OpenCV-native format), but YOLO/Torch expect RGB image in float format
  frame.copyTo(img);
  
  cv::resize(img, img, cv::Size(szml, szml),cv::InterpolationFlags::INTER_CUBIC);

  if(img.channels() == 1) {
    cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);
  } else {
    assert(img.type() == CV_8UC3 && "Color input frames in the BGR (OpenCV-native) format is expected");
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  }
  img.convertTo(img, CV_32FC3, 1.0f / 255.0f);
  auto input_tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 3});
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
            float left = det[i][0].item().toFloat() * img.cols / szml;
            float top = det[i][1].item().toFloat() * img.rows / szml;
            //float right = det[i][2].item().toFloat() * img.cols / szml;
            //float bottom = det[i][3].item().toFloat() * img.rows / szml;
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
