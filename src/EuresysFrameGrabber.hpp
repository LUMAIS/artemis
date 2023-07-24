#pragma once

#include "FrameGrabber.hpp"

#include <EGrabber.h>

#include "Options.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <mutex>


namespace fort {
namespace artemis {

class EuresysFrame : public Frame,public Euresys::ScopedBuffer {
public :
	EuresysFrame(Euresys::EGrabber<Euresys::CallbackOnDemand> & grabber,
				 const Euresys::NewBufferData &,
				 uint64_t & lastFrame,
				 uint64_t & toAdd,
				 std::string CameraID, uint64_t EventCount);

	virtual ~EuresysFrame();

	virtual void * Data();
	virtual size_t Width() const;
	virtual size_t Height() const;
	virtual std::string CameraID() const;
	virtual uint64_t EventCount() const;
	virtual uint64_t Timestamp() const;
	virtual uint64_t ID() const;
	const cv::Mat & ToCV();
private :
	size_t d_width,d_height;
	std::string d_cameraid;
	uint64_t d_timestamp,d_ID,d_eventcount;
	cv::Mat d_mat;
	friend class EuresysFrameGrabbero;
};


class EuresysFrameGrabber : public FrameGrabber,public Euresys::EGrabber<Euresys::CallbackOnDemand>{//, public Euresys::EGrabberCallbackSingleThread{
public :
	typedef std::shared_ptr<Euresys::ScopedBuffer> BufferPtr;

	EuresysFrameGrabber(Euresys::EGenTL & gentl,
	                    const CameraOptions & options, int & interfaceIndex, int & deviceIndex);

	virtual ~EuresysFrameGrabber();


	void Start() override;
	void Stop() override;
	Frame::Ptr NextFrame() override;
	//boot MatchingCameraID() override;

	cv::Size Resolution() const override;
private:

	virtual void onNewBufferEvent(const Euresys::NewBufferData &data);

	std::mutex         d_mutex;
	Frame::Ptr         d_frame;
	uint64_t           d_lastFrame;
	uint64_t           d_toAdd;
	int32_t            d_width,d_height;
	std::string        d_cameraid;
	uint64_t		   d_eventcount;
};


} // namespace artemis

} // namespace fort
