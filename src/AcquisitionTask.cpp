#include "AcquisitionTask.hpp"

#include <artemis-config.h>
#ifndef FORCE_STUB_FRAMEGRABBER_ONLY
#include "EuresysFrameGrabber.hpp"
#endif //FORCE_STUB_FRAMEGRABBER_ONLY
#include "StubFrameGrabber.hpp"
#include "StubVideoGrabber.hpp"

#include "ProcessFrameTask.hpp"

#include <glog/logging.h>


namespace fort {
namespace artemis {

FrameGrabber::Ptr AcquisitionTask::LoadFrameGrabber(const std::vector<std::string> & stubImagePaths, std::string inputVideoPath, 
                                                    const CameraOptions & options) {
#ifndef FORCE_STUB_FRAMEGRABBER_ONLY
	if (stubImagePaths.empty() && inputVideoPath.length() == 0) {
		static Euresys::EGenTL egentl;	

	using namespace Euresys;

	int interfaceID = 0;
	int deviceID = 0;

	//--Serhii--8.10.2021
		gc::TL_HANDLE tl = egentl.tlOpen();
    	uint32_t numInterfaces = egentl.tlGetNumInterfaces(tl);
		LOG(INFO) << "[LoadFrameGrabber]:  numInterfaces - "<<numInterfaces;

		for (uint32_t interfaceIndex = 0; interfaceIndex < numInterfaces; interfaceIndex++) 
		{
        	std::string interfaceID = egentl.tlGetInterfaceID(tl, interfaceIndex);
        	gc::IF_HANDLE interfaceHandle = egentl.tlOpenInterface(tl, interfaceID);
        	uint32_t numDevice = egentl.ifGetNumDevices(interfaceHandle);
			LOG(INFO) << "[LoadFrameGrabber]:  numDevice - "<<numDevice;

			for (uint32_t deviceIndex = 0; deviceIndex < numDevice; deviceIndex++) 
			{
            	std::string deviceIDstring = egentl.ifGetDeviceID(interfaceHandle, deviceIndex);
            	gc::DEV_HANDLE deviceHandle = egentl.ifOpenDevice(interfaceHandle, deviceIDstring);

            	LOG(INFO) << "[LoadFrameGrabber]:  deviceIndex - "<<deviceIndex;

				try 
				{
                	if (egentl.devGetPort(deviceHandle)) 
					{
                    	//grabbers.push_back(new MyGrabber(genTL, interfaceIndex, deviceIndex, interfaceID, deviceID));
						//LOG(INFO) << "[LoadFrameGrabber]: Camera connected OK "<<interfaceID<<" <"<<deviceID<<">"<<std::endl;
                	}
            	} 
				catch (const Euresys::gentl_error &) {
					LOG(INFO) << "[LoadFrameGrabber]: no camera connected on "<<interfaceID<<" <"<<deviceID<<">"<<std::endl;
            	}
				
				if(std::stoi(options.cameraID) == deviceIndex)
				{
					interfaceID = interfaceIndex;
					deviceID = deviceIndex;
					interfaceIndex = numInterfaces;
					deviceIndex = numDevice;
					LOG(INFO) << "[LoadFrameGrabber]: Camera selected "<<deviceID<<std::endl;
					
					break;
				}
        	}
		}
	//--Serhii--8.10.2021
		return std::make_shared<EuresysFrameGrabber>(egentl,options,interfaceID,deviceID);
	} else {
		if(!stubImagePaths.empty())
		{
			//return std::make_shared<StubFrameGrabber>(stubImagePaths,options.FPS);
			return std::make_shared<StubFrameGrabber>(stubImagePaths,options);
		}
		else
		{
			//return std::make_shared<StubVideoGrabber>(inputVideoPath,options.FPS);
			return std::make_shared<StubVideoGrabber>(inputVideoPath,options);
		}
			
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
	bool ptrframe = true;

	while(d_quit.load() == false && ptrframe == true) {
		Frame::Ptr f = d_grabber->NextFrame();

		if(!f) ptrframe = false;

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
	sleep(5);
	raise(SIGINT);
}


} // namespace artemis
} // namespace fort
