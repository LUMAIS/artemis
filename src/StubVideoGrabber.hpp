#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>

#include "FrameGrabber.hpp"
#include "Options.hpp"

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
    StubVideoGrabber(const std::string & path,
                     const CameraOptions & options);//double FPS);

    virtual ~StubVideoGrabber();

    void Start() override;
    void Stop() override;
    Frame::Ptr NextFrame() override;

    cv::Size Resolution() const override;
protected:
    void captureFrame();
private:
    typedef std::chrono::high_resolution_clock clock;
    typedef clock::time_point time;
    cv::Mat              d_frame;
    uint64_t             d_ID, d_timestamp, d_eventcount;
    Time                 d_last;
    Duration             d_period;
    std::string			 d_cameraid;

    cv::VideoCapture     d_cap;
    mutable bool         d_convFrame;  //! Whether the frame format conversion is required
};

} // namespace artemis
} // namespace fort
