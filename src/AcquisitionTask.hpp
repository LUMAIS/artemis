#pragma once

#include "Task.hpp"

#include "FrameGrabber.hpp"

#include "Options.hpp"

#include <atomic>

namespace fort {
namespace artemis {

class ProcessFrameTask;
typedef std::shared_ptr<ProcessFrameTask> ProcessFrameTaskPtr;



class AcquisitionTask : public Task {
public:

	static FrameGrabber::Ptr LoadFrameGrabber(const std::vector<std::string> & stubImagePaths, std::string inputVideoPath,
	                                          const CameraOptions & options, const VideoOutputOptions & vidOpts, std::string cameraID = "0");

	AcquisitionTask(const FrameGrabber::Ptr & grabber,
	                const ProcessFrameTaskPtr &  process);

	virtual ~AcquisitionTask();

	void Run() override;

	void Stop();

	ProcessFrameTaskPtr& process();

private:

	FrameGrabber::Ptr   d_grabber;
	ProcessFrameTaskPtr d_processFrame;
	std::atomic<bool>   d_quit;
	
};

} // namespace artemis
} // namespace fort
