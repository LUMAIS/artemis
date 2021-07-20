#pragma once

#include <opencv2/core/core.hpp>

#include "FrameGrabber.hpp"

namespace fort {
namespace artemis {

/*
class StubFrame : public Frame {
public :
	StubFrame(const cv::Mat & mat, uint64_t ID);
	virtual ~StubFrame();


	virtual void * Data();
	virtual size_t Width() const;
	virtual size_t Height() const;
	virtual uint64_t Timestamp() const;
	virtual uint64_t ID() const;
	const cv::Mat & ToCV();
private :
	uint64_t d_ID;
	cv::Mat d_mat;
};
*/

class StubVideoGrabber : public FrameGrabber {
public :
	StubVideoGrabber(const std::string & paths,
	                 double FPS);

	virtual ~StubVideoGrabber();

	void Start() override;
	void Stop() override;
	Frame::Ptr NextFrame() override;

	cv::Size Resolution() const override;
private:
	typedef std::chrono::high_resolution_clock clock;
	typedef clock::time_point time;
	std::vector<cv::Mat> d_images;
	uint64_t             d_ID,d_timestamp;
	Time                 d_last;
	Duration             d_period;
};


} // namespace artemis
} // namespace fort
