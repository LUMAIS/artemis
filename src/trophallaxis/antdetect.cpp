#include "antdetect.hpp"

#include <unistd.h>

/*std::vector<std::array<float,2>> TDetect(bool useCUDA, std::string modelFilepath, std::string labelFilepath, cv::Mat imageBGR, size_t nThreads)
{
    int koef = 2;
    std::string instanceName{"image-classification-inference"};

    Ort::SessionOptions sessionOptions;
    
    std::vector<std::string> labels{readLabels(labelFilepath)};

    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
    
    sessionOptions.SetIntraOpNumThreads(1);

    if (useCUDA)
    {
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
    }

    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session session(env, modelFilepath.c_str(), sessionOptions);

    //cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
//---------------------------------------------------------------------

    //v::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    cv::Mat resizedImageBGR;
    cv::Mat bufimageBGR;
    cv::Mat img;
    //cv::resize(imageBGR, resizedImageBGR,cv::Size(500, 500),cv::InterpolationFlags::INTER_CUBIC);

    cv::Point2f pd;
    cv::Point2f p1;
    

    int step = 30;

    int startX = 0;
    int startY = 0;
    //p1.x = 166;
    //p1.y = 565;

    p1.x = 0;
    p1.y = 0;

    bufimageBGR = resizedImageBGR;

    float d;

    std::vector<cv::Point2f> detects;

    int test = 0;
    while(p1.y + step < 1000 && test < 100)
    {
        test++;
        img = imageBGR(cv::Rect(p1.x*4, p1.y*4, 400, 400));
        d = detect(img,labels,session);
        if(d >= 0.99)
        {
            pd.x = (p1.x+50)/koef;
            pd.y = (p1.y+50)/koef;

            detects.push_back(pd);
        }
        
        if(d > 0.69)
            step = 3;

        if(d < 0.1)
            step = 20;

        cv::resize(imageBGR, bufimageBGR,cv::Size(1000/koef, 1000/koef),cv::InterpolationFlags::INTER_CUBIC);

        drawrec(bufimageBGR,p1,d,koef);

        for(int j=0; j<detects.size(); j++)
        {
            cv::circle(bufimageBGR, detects.at(j), 5, cv::Scalar(0, 0, 255), 10/koef);
        }

        imshow("Detector", bufimageBGR);
        cv::waitKey(100);

        p1.x += step;
        if(p1.x + 100 > 1000)
        {
            p1.x = 0;
            p1.y += step;
        }
    }    

    std::vector<std::array<float,2>> v;
	std::array<float,2> p;
	p[0] = 15;
	p[1] = 18;
	v.push_back(p);
	return v;
}
*/

/*std::vector<cv::Point2f> detectorT (torch::jit::script::Module module, cv::Mat frame, torch::DeviceType device_type)
{ 
  std::vector<std::string> labels;
  labels.push_back("more than one ant");
  labels.push_back("no ants");
  labels.push_back("one ant");
  labels.push_back("trophallaxis");

  int pinput = 175;
  int step = 100;

  cv::Point2f pd;
  cv::Point2f p1;
  p1.x = 0;
  p1.y = 0;

  cv::Mat imageBGR;
  cv::Mat bufimageBGR;

  std::vector<cv::Point2f> detects;
  std::vector<cv::Point2f> detectsforcircle;
  
  while(p1.y + pinput < frame.rows)
  {
    imageBGR = frame(cv::Rect(p1.x, p1.y, pinput, pinput));
    //cv::resize(imageBGR, imageBGR, cv::Size(pinput, pinput), cv::INTER_CUBIC);
    cv::cvtColor(imageBGR, imageBGR, cv::COLOR_BGR2RGB);
    imageBGR.convertTo(imageBGR, CV_32FC3, 1.0f / 255.0f);
    auto input_tensor = torch::from_blob(imageBGR.data, {1, imageBGR.rows, imageBGR.cols, 3});
    input_tensor = input_tensor.permute({0, 3, 1, 2}).contiguous();
    
    input_tensor = input_tensor.to(device_type);
    std::vector<torch::jit::IValue> input;
    input.emplace_back(input_tensor);

    at::Tensor output = module.forward(input).toTensor();
    //std::cout << labels[output.argmax(1).item().toInt()] << '\n';

    if(labels[output.argmax(1).item().toInt()] == "trophallaxis")
    {
      pd.x = p1.x+pinput/2;
      pd.y = p1.y+pinput/2;
      detects.push_back(pd);
    }

    p1.x += step;
    if(p1.x + pinput >= frame.cols)
    {
        p1.x = 0;
        p1.y += step;
    }
  }

  return detects;
}
*/

std::vector<cv::Point2f> detectorT (torch::jit::script::Module module, cv::Mat frame, torch::DeviceType device_type)
{
  int resolution = 992;
  int pointsdelta = 30;
  std::vector<cv::Point2f> detects;
  std::vector<cv::Point2f> detectsbuf;
  cv::Mat imageBGR;
  cv::resize(frame, imageBGR,cv::Size(992, 992),cv::InterpolationFlags::INTER_CUBIC);

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


