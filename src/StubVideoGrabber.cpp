#include "StubVideoGrabber.hpp"
#include "StubFrameGrabber.hpp"

#include <thread>

#include <opencv2/highgui/highgui.hpp>

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include <unistd.h>

#include <glog/logging.h>

#include "FrameGrabber.hpp"

namespace fort {
namespace artemis {


/*
std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}
*/

StubVideoGrabber::StubVideoGrabber(const std::string & paths,
                                   double FPS)
	: d_ID(0)
	, d_timestamp(0)
	, d_period(1.0e9 / FPS) {
	if ( paths.empty() == true ) {
		throw std::invalid_argument("No path given to StubVideoGrabber");
	}

	LOG(INFO) << "[StubVideoGrabber]: Loading Video "<<paths;

	cv::VideoCapture cap(paths);

     if (!cap.isOpened()) {
			LOG(INFO) << "[StubVideoGrabber]: Cannot open the video file";
     }
	 else
	 {
		cv::Mat framebuf;
		for (int frame_count = 0; frame_count < cap.get(cv::CAP_PROP_FRAME_COUNT); frame_count++) 
		{
        	cap.set(cv::CAP_PROP_POS_FRAMES, frame_count);
          	

          	if (!cap.read(framebuf)) {
				LOG(INFO) << "[StubVideoGrabber]: Failed to extract the frame "<<frame_count;
          	}
			else
			{
				//std::string ty =  type2str(framebuf.type());
				//printf("Matrix 1: %s(%d) %dx%d \n", ty.c_str(),framebuf.type(), framebuf.cols, framebuf.rows );
				
				cv::Mat frame;

				cv::cvtColor(framebuf, frame, cv::COLOR_RGB2GRAY);
				
				d_images.push_back(frame);

				LOG(INFO) << "[StubVideoGrabber]: Success to extracted the frame "<<frame_count;
			}
    	}
	 }
}

StubVideoGrabber::~StubVideoGrabber() {
}



void StubVideoGrabber::Start() {
	d_last = Time::Now().Add(-d_period);
}

void StubVideoGrabber::Stop() {
}

cv::Size StubVideoGrabber::Resolution() const {
	return d_images.front().size();
}

Frame::Ptr StubVideoGrabber::NextFrame() {
	auto toWait = d_last.Add(d_period).Sub(Time::Now());
	if ( toWait > 0 ) {
		usleep(toWait.Microseconds());
	}

	Frame::Ptr res = std::make_shared<StubFrame>(d_images[d_ID % d_images.size()],d_ID);

	d_ID += 1;
	d_last = res->Time();

	//printf("d_images.size() = %ld | d_ID: %ld\n",d_images.size(), d_ID);

	if(d_ID <= d_images.size()+1)
		return res;
	else
		return nullptr;
}

} // namespace artemis
} // namespace fort
