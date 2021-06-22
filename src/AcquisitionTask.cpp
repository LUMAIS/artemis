#include "AcquisitionTask.hpp"

#include <artemis-config.h>
#ifndef FORCE_STUB_FRAMEGRABBER_ONLY
#include "EuresysFrameGrabber.hpp"
#endif //FORCE_STUB_FRAMEGRABBER_ONLY
#include "StubFrameGrabber.hpp"

#include "ProcessFrameTask.hpp"

#include <glog/logging.h>

namespace fort {
namespace artemis {

FrameGrabber::Ptr AcquisitionTask::LoadFrameGrabber(const std::vector<std::string> & stubImagePaths,
                                                    const CameraOptions & options) {
#ifndef FORCE_STUB_FRAMEGRABBER_ONLY
	if (stubImagePaths.empty() ) {
		static Euresys::EGenTL egentl;
		return std::make_shared<EuresysFrameGrabber>(egentl,options);
	} else {
		return std::make_shared<StubFrameGrabber>(stubImagePaths,options.FPS);
	}
#else
	return std::make_shared<StubFrameGrabber>(stubImagePaths,options.FPS);
#endif
}


AcquisitionTask::AcquisitionTask(const FrameGrabber::Ptr & grabber,
                                 const ProcessFrameTaskPtr &  process)
	: d_grabber(grabber)
	, d_processFrame(process) {
	d_quit.store(false);
}

AcquisitionTask::~AcquisitionTask() { }

void AcquisitionTask::Stop() {
	d_quit.store(true);
}

void AcquisitionTask::Run() {
	LOG(INFO) << "[AcquisitionTask]:  started";
	d_grabber->Start();
	while(d_quit.load() == false) {
		Frame::Ptr f = d_grabber->NextFrame();
		if ( d_processFrame ) {
			d_processFrame->QueueFrame(f);
		}
	}
	LOG(INFO) << "[AcquisitionTask]:  Tear Down";
	d_grabber->Stop();
	if (d_processFrame) {
		d_processFrame->CloseFrameQueue();
	}
	LOG(INFO) << "[AcquisitionTask]:  ended";
}


} // namespace artemis
} // namespace fort
