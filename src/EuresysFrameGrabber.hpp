#pragma once

#include "FrameGrabber.hpp"

#include <EGrabber.h>

#include "Options.hpp"

#include <opencv2/core/core.hpp>
#include <mutex>


namespace fort {
namespace artemis {

class EuresysFrame : public Frame,public Euresys::ScopedBuffer {
public :
	EuresysFrame(Euresys::EGrabber<Euresys::CallbackOnDemand> & grabber,
				 const Euresys::NewBufferData &,
				 uint64_t & lastFrame,
				 uint64_t & toAdd,
				 size_t & RenderHeight,
				 std::string CameraID);

	virtual ~EuresysFrame();

	virtual void * Data();
	virtual size_t Width() const;
	virtual size_t Height() const;
	virtual std::string CameraID() const;
	virtual size_t RenderHeight() const;
	virtual uint64_t Timestamp() const;
	virtual uint64_t ID() const;
	const cv::Mat & ToCV();
private :
	size_t d_width,d_height;
	size_t d_renderheight;
	std::string d_cameraid;
	uint64_t d_timestamp,d_ID;
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

	cv::Size Resolution() const override;
private:

	virtual void onNewBufferEvent(const Euresys::NewBufferData &data);

	std::mutex         d_mutex;
	Frame::Ptr         d_frame;
	uint64_t           d_lastFrame;
	uint64_t           d_toAdd;
	int32_t            d_width,d_height;
	size_t			   d_renderheight;
	std::string        d_cameraid;
};


} // namespace artemis

} // namespace fort
