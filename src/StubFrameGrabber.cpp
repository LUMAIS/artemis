#include "StubFrameGrabber.hpp"

#include <thread>

#include <opencv2/highgui/highgui.hpp>

#include <unistd.h>

#include <glog/logging.h>

namespace fort {
namespace artemis {

StubFrame::StubFrame(const cv::Mat & mat, uint64_t ID, std::string CameraID)
	: d_mat(mat.clone())
	, d_ID(ID)
	, d_cameraid(CameraID)
{
}

StubFrame::~StubFrame() {}

void * StubFrame::Data() {
	return d_mat.data;
}

size_t StubFrame::Width() const {
	return d_mat.cols;
}

size_t StubFrame::Height() const {
	return d_mat.rows;
}

std::string StubFrame::CameraID() const {
	return d_cameraid;
}

uint64_t StubFrame::EventCount() const {
	return d_eventcount;
}

uint64_t StubFrame::Timestamp() const {
	return Time().MonotonicValue() / 1000;
}

uint64_t StubFrame::ID() const {
	return d_ID;
}

const cv::Mat & StubFrame::ToCV() {
	return d_mat;
}


StubFrameGrabber::StubFrameGrabber(const std::vector<std::string> & paths,
                                 const CameraOptions & options)  //double FPS)
	: d_ID(0)
	, d_timestamp(0)
	, d_period(1.0e9 / options.FPS)
	, d_cameraid(options.cameraID) 
{

	if ( paths.empty() )
		throw std::invalid_argument("No paths given to StubFrameGrabber");

	for ( const auto & p : paths ) {
#ifdef HAVE_OPENCV_CUDACODEC
#else
#endif // HAVE_OPENCV_CUDACODEC
		d_images.push_back(cv::imread(p, cv::IMREAD_GRAYSCALE));

		LOG(INFO) << "[StubFrameGrabber]: Loading frame "<<p;

		if ( d_images.back().data == NULL ) {
			throw std::runtime_error("Could not load '" + p + "'");
		}
		if ( d_images.back().size() != d_images[0].size() ) {
			throw std::runtime_error("'" + paths[0] + "' and '" + p + "' have different sizes");
		}
	}
}

StubFrameGrabber::~StubFrameGrabber() {
}

void StubFrameGrabber::Start() {
	d_last = Time::Now().Add(-d_period);
}

void StubFrameGrabber::Stop() {
}

cv::Size StubFrameGrabber::Resolution() const {
	return d_images.front().size();
}

Frame::Ptr StubFrameGrabber::NextFrame() {
	auto toWait = d_last.Add(d_period).Sub(Time::Now());
	if ( toWait > 0 ) {
		usleep(toWait.Microseconds());
	}

	Frame::Ptr res = std::make_shared<StubFrame>(d_images[d_ID % d_images.size()],d_ID,d_cameraid);
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
