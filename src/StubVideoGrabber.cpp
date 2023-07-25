#include <thread>

#include <opencv2/imgproc.hpp>

#include <unistd.h>

#include <glog/logging.h>

#include "StubFrameGrabber.hpp"
#include "StubVideoGrabber.hpp"

namespace fort {
namespace artemis {

StubVideoGrabber::StubVideoGrabber(const std::string & path,
                                   const CameraOptions & options)//double FPS)
	: d_ID(0)
	, d_timestamp(0)
	, d_period(1.0e9 / options.FPS)
	, d_cameraid(options.cameraID) 
{
	if (path.empty())
		throw std::invalid_argument("No path given to StubVideoGrabber");
	LOG(INFO) << "[StubVideoGrabber]: Loading Video: " << path;

	d_cap = cv::VideoCapture(path);
	if (!d_cap.isOpened()) {
		LOG(INFO) << "[StubVideoGrabber]: Cannot open the video file";
		throw std::runtime_error("Cannot open the video file");
	}

	// Note: video processing expects grayscale frames, so it is better to convert the frames at once if necessary
	d_convFrame = d_cap.get(cv::CAP_PROP_FORMAT) != CV_8UC1;
	// Ensure that !d_convFrame is actual
	if(!d_convFrame) {
		cv::Mat framebuf;
		if (d_cap.read(framebuf)) {
			LOG(INFO) << "[StubVideoGrabber]: color video: " << d_convFrame << ", framebuf channels: " << framebuf.channels() << ", framebuf type: " << framebuf.type();

			if(framebuf.channels() > 1) {
				d_convFrame = true;
				cv::cvtColor(framebuf, framebuf, cv::COLOR_BGR2GRAY);
				LOG(INFO) << "[StubVideoGrabber]: Converted framebuf channels: " << framebuf.channels() << ", framebuf type: " << framebuf.type();
			}
			assert(framebuf.channels() == 1 && "Grayscale input video frames are expected");

			// Restart capturing from the first frame
			d_cap.set(cv::CAP_PROP_POS_FRAMES, 0);
		} else LOG(WARNING) << "[StubVideoGrabber]: Failed to extract the initial frame";
	}

	// Capture the first frame
	captureFrame();

	LOG(INFO) << "[StubVideoGrabber]: The first video frame is extracted successfully";
}

StubVideoGrabber::~StubVideoGrabber() {
}

void StubVideoGrabber::Start() {
	d_last = Time::Now().Add(-d_period);
}

void StubVideoGrabber::Stop() {
}

cv::Size StubVideoGrabber::Resolution() const {
	return d_frame.size();
}

Frame::Ptr StubVideoGrabber::NextFrame() {
	if(d_frame.empty())
		return nullptr;

	const auto toWait = d_last.Add(d_period).Sub(Time::Now());
	if (toWait > 0)
		usleep(toWait.Microseconds());

	Frame::Ptr res = std::make_shared<StubFrame>(d_frame, d_ID, d_cameraid);
	d_ID += 1;
	d_last = res->Time();

	// Capture the following frame
	captureFrame();

	return res;
}

void StubVideoGrabber::captureFrame()
{
	cv::Mat framebuf;

	// Note: even if the frame is not extracted, update d_frame (to empty value)
#ifdef HAVE_OPENCV_CUDACODEC
#else
#endif // HAVE_OPENCV_CUDACODEC
	if (!d_cap.read(framebuf))
		LOG(WARNING) << "[StubVideoGrabber]: Failed to extract the frame: " << d_ID;
	else if (d_convFrame) {
		// assert(framebuf.channels() == 3 && "3-channel color frame is expected");
		cv::cvtColor(framebuf, framebuf, cv::COLOR_BGR2GRAY);
	}

	d_frame = framebuf;

	// // Make larger square frame
	// const uint16_t sz = framebuf.rows > framebuf.cols
	// 	? framebuf.rows : framebuf.cols;
	// cv::Mat frame(sz, sz, CV_8UC1, cv::Scalar(0));
	// if(framebuf.rows > framebuf.cols)
	// 	framebuf.copyTo(frame(cv::Rect((framebuf.rows - framebuf.cols)/2, 0, framebuf.cols, framebuf.rows)));
	// else
	// 	framebuf.copyTo(frame(cv::Rect(0, (framebuf.cols - framebuf.rows)/2, framebuf.cols, framebuf.rows)));
	// d_frame = frame;
	
	// LOG(INFO) << "[StubVideoGrabber]: extract frame: " << d_ID;
}

} // namespace artemis
} // namespace fort
