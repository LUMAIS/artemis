#pragma once

#include <opencv2/core/core.hpp>

#include "FrameGrabber.hpp"
#include "Options.hpp"

namespace fort {
namespace artemis {

class StubFrame : public Frame {
public :
	StubFrame(const cv::Mat & mat, uint64_t ID, std::string CameraID, size_t RenderHeight);
	virtual ~StubFrame();

	virtual void * Data();
	virtual size_t Width() const;
	virtual size_t Height() const;
	virtual std::string CameraID() const;
	virtual uint64_t EventCount() const;
	virtual size_t RenderHeight() const;
	virtual uint64_t Timestamp() const;
	virtual uint64_t ID() const;
	const cv::Mat & ToCV();
private :
	uint64_t d_ID,d_eventcount;
	cv::Mat d_mat;
	std::string d_cameraid;
	size_t d_renderheight;
};


class StubFrameGrabber : public FrameGrabber {
public :
	StubFrameGrabber(const std::vector<std::string> & paths,
	                 const CameraOptions & options);//double FPS);

	virtual ~StubFrameGrabber();

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
	std::string 		 d_cameraid;
	size_t				 d_renderheight;
};


} // namespace artemis
} // namespace fort
