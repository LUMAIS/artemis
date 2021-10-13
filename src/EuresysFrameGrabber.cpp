#include "EuresysFrameGrabber.hpp"

#include <glog/logging.h>
#include <regex>

namespace fort {
namespace artemis {


EuresysFrameGrabber::EuresysFrameGrabber(Euresys::EGenTL & gentl,
                                         const CameraOptions & options)
	: Euresys::EGrabber<Euresys::CallbackOnDemand>(gentl)
	, d_lastFrame(0)
	, d_toAdd(0)
	, d_width(0)
	, d_height(0) 
	, d_cameraid(0)
	, d_renderheight(0){

	using namespace Euresys;

	//--Serhii--8.10.2021
	/*
		gc::TL_HANDLE tl = gentl.tlOpen();
    	uint32_t numInterfaces = gentl.tlGetNumInterfaces(tl);
		LOG(INFO) << "[LoadFrameGrabber]:  numInterfaces - "<<numInterfaces;

		for (uint32_t interfaceIndex = 0; interfaceIndex < numInterfaces; interfaceIndex++) 
		{
        	std::string interfaceID = gentl.tlGetInterfaceID(tl, interfaceIndex);
        	gc::IF_HANDLE interfaceHandle = gentl.tlOpenInterface(tl, interfaceID);
        	uint32_t numDevice = gentl.ifGetNumDevices(interfaceHandle);

			numDevice = 1;
			LOG(INFO) << "[LoadFrameGrabber]:  numDevice - "<<numDevice;

			for (uint32_t deviceIndex = 0; deviceIndex < numDevice; deviceIndex++) 
			{
            	std::string deviceID = gentl.ifGetDeviceID(interfaceHandle, deviceIndex);
            	gc::DEV_HANDLE deviceHandle = gentl.ifOpenDevice(interfaceHandle, deviceID);

            	LOG(INFO) << "[LoadFrameGrabber]:  deviceIndex - "<<deviceIndex;

				try 
				{
                	if (gentl.devGetPort(deviceHandle)) 
					{
                    	//grabbers.push_back(new MyGrabber(genTL, interfaceIndex, deviceIndex, interfaceID, deviceID));
						LOG(INFO) << "[LoadFrameGrabber]: Camera connected OK "<<interfaceID<<" <"<<deviceID<<">"<<std::endl;
                	}
            	} 
				catch (const Euresys::gentl_error &) {
					LOG(INFO) << "[LoadFrameGrabber]: no camera connected on "<<interfaceID<<" <"<<deviceID<<">"<<std::endl;
            	}
        	}
		}
	//--Serhii--8.10.2021*/

	std::string ifID = getString<InterfaceModule>("InterfaceID");
	std::regex slaveRx("df-camera");
	bool isMaster = !std::regex_search(ifID,slaveRx);

	if ( isMaster == true ) {
		d_width = getInteger<RemoteModule>("Width");
		d_height = getInteger<RemoteModule>("Height");

		DLOG(INFO) << "Width: "<<d_width;
		DLOG(INFO) << "Height: "<<d_height;

		DLOG(INFO) << "LineSelector: IOUT11";
		setString<InterfaceModule>("LineSelector","IOUT11");
		DLOG(INFO) << "LineInverter: True";
		setString<InterfaceModule>("LineInverter","True");

		std::string DeviceIDstr = "Device0Strobe";
		if(!options.cameraID.empty())
		{
			
			if(std::stoi(options.cameraID) > -1)
			{
				DeviceIDstr = "Device" + options.cameraID + "Strobe";
				d_cameraid = options.cameraID;
			}
		}

		DLOG(INFO) << DeviceIDstr;
		setString<InterfaceModule>("LineSource",DeviceIDstr);

		//DLOG(INFO) << "LineSource: Device0Strobe";
		//setString<InterfaceModule>("LineSource","Device0Strobe");

		DLOG(INFO) << "CameraControlMethod: RC";
		setString<DeviceModule>("CameraControlMethod","RC");

		//This is a big hack allowing to have the camera controlled by the
		//framegrabber. We set it to pulse mode and double the frequency.
		setString<DeviceModule>("ExposureReadoutOverlap","True");
		DLOG(INFO) << "AcquisitionFrameRate: " << options.FPS;
		setInteger<DeviceModule>("CycleMinimumPeriod",1e6/(2*options.FPS));

		setString<DeviceModule>("CxpTriggerMessageFormat","Toggle");

		setInteger<DeviceModule>("ExposureTime",6000);
		DLOG(INFO) << "StrobeDuration: " << options.StrobeDuration;
		setInteger<DeviceModule>("StrobeDuration",options.StrobeDuration.Microseconds());

		DLOG(INFO) << "StrobeDelay: " << options.StrobeDelay;
		setInteger<DeviceModule>("StrobeDelay",options.StrobeDelay.Microseconds());

		//Serhii--9.10.2021 setString<RemoteModule>("ExposureMode","Edge_Triggerred_Programmable");

	} else {
		if (options.SlaveWidth == 0 || options.SlaveHeight == 0 ) {
			throw std::runtime_error("Camera resolution is not specified in DF mode");
		}

		setInteger<RemoteModule>("Width",options.SlaveWidth);
		setInteger<RemoteModule>("Height",options.SlaveHeight);
		d_width = options.SlaveWidth;
		d_height = options.SlaveHeight;
	}
	
	d_renderheight = d_height;

	if(!options.cameraID.empty())
	{
		if(options.RenderHeight > 0)
			d_renderheight = options.RenderHeight;
	
	}

	
	DLOG(INFO) << "Enable Event";
	enableEvent<NewBufferData>();
	DLOG(INFO) << "Realloc Buffer";
	reallocBuffers(4);
}

void EuresysFrameGrabber::Start() {
	DLOG(INFO) << "Starting framegrabber";
	start();
}

void EuresysFrameGrabber::Stop() {
	DLOG(INFO) << "Stopping framegrabber";
	stop();
	DLOG(INFO) << "/*Framegrabber stopped";
}


EuresysFrameGrabber::~EuresysFrameGrabber() {
	DLOG(INFO) << "Cleaning FrameGrabber";
}

cv::Size EuresysFrameGrabber::Resolution() const {
	return cv::Size(d_width,d_height);
}

Frame::Ptr EuresysFrameGrabber::NextFrame() {
	processEvent<Euresys::NewBufferData>(1000);
	Frame::Ptr res = d_frame;
	d_frame.reset();
	return res;
}

void EuresysFrameGrabber::onNewBufferEvent(const Euresys::NewBufferData &data) {
	std::unique_lock<std::mutex> lock(d_mutex);
	d_frame = std::make_shared<EuresysFrame>(*this,data,d_lastFrame,d_toAdd);
}

EuresysFrame::EuresysFrame(Euresys::EGrabber<Euresys::CallbackOnDemand> & grabber, const Euresys::NewBufferData & data, uint64_t & lastFrame, uint64_t & toAdd )
	: Euresys::ScopedBuffer(grabber,data)
	, d_width(getInfo<size_t>(GenTL::BUFFER_INFO_WIDTH))
	, d_height(getInfo<size_t>(GenTL::BUFFER_INFO_HEIGHT))
	, d_timestamp(getInfo<uint64_t>(GenTL::BUFFER_INFO_TIMESTAMP))
	, d_ID(getInfo<uint64_t>(GenTL::BUFFER_INFO_FRAMEID))
	, d_mat(d_height,d_width,CV_8U,getInfo<void*>(GenTL::BUFFER_INFO_BASE)) {
	if ( d_ID == 0 && lastFrame != 0 ) {
		toAdd += lastFrame + 1;
	}
	lastFrame = d_ID;
	d_ID += toAdd;
}

EuresysFrame::~EuresysFrame() {}

size_t EuresysFrame::Width() const {
	return d_width;
}
size_t EuresysFrame::Height() const {
	return d_height;
}

std::string EuresysFrame::CameraID() const {
	return d_cameraid;
}

size_t EuresysFrame::RenderHeight() const {
	return d_renderheight;
}

uint64_t EuresysFrame::Timestamp() const {
	return d_timestamp;
}

uint64_t EuresysFrame::ID() const {
	return d_ID;
}

const cv::Mat & EuresysFrame::ToCV() {
	return d_mat;
}

void * EuresysFrame::Data() {

	return getInfo<void*>(GenTL::BUFFER_INFO_BASE);
}

} // namespace artemis
} // namespace fort
