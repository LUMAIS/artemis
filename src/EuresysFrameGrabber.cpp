#include "EuresysFrameGrabber.hpp"

#include <glog/logging.h>
#include <regex>
#include <fstream>

namespace fort
{
	namespace artemis
	{

		EuresysFrameGrabber::EuresysFrameGrabber(Euresys::EGenTL &gentl,
												 const CameraOptions &options, int &interfaceIndex, int &deviceIndex)
			: Euresys::EGrabber<Euresys::CallbackOnDemand>(gentl, interfaceIndex, deviceIndex), d_lastFrame(0), d_toAdd(0), d_width(0), d_height(0), d_cameraid("0"), d_eventcount(0)
		{

			using namespace Euresys;

			std::string ifID = getString<InterfaceModule>("InterfaceID");
			std::regex slaveRx("df-camera");
			bool isMaster = !std::regex_search(ifID, slaveRx);

			if (isMaster == true)
			{
				std::string DeviceIDstr;

				if (options.Triggermode != "none")
				{
					DeviceIDstr = "Device" + std::to_string(deviceIndex) + "Strobe";
					d_cameraid = std::to_string(deviceIndex);
				}
				else
				{
					DeviceIDstr = "Device" + options.cameraID + "Strobe";
					if (!options.cameraID.empty())
					{
						if (std::stoi(options.cameraID) > -1)
						{
							DeviceIDstr = "Device" + options.cameraID + "Strobe";
							d_cameraid = options.cameraID;
						}
					}
				}

				// This is a big hack allowing to have the camera controlled by the
				// framegrabber. We set it to pulse mode and double the frequency.

				// Serhii--18.12.2021--
				std::string DeviceModelName;
				DeviceModelName = getString<RemoteModule>("DeviceModelName");
				DLOG(INFO) << "DeviceModelName - " + DeviceModelName;

				if (options.Triggermode != "none")
				{
					std::string config = "../configs/" + DeviceModelName + ".js";

					try
					{
						runScript(config);
						DLOG(INFO) << "Camera Configuration - OK!";
						// cv::Size sizeFrame(testimg.cols, testimg.rows);
					}
					catch (...)
					{
						DLOG(INFO) << "Failed to configure camera, file (" + config + ") is missing or camera not available";
					}
				}
				else
				{
					DLOG(INFO) << "LineSelector: IOUT11";
					setString<InterfaceModule>("LineSelector", "IOUT11");

					DLOG(INFO) << "LineInverter: True";
					setString<InterfaceModule>("LineInverter", "True");

					DLOG(INFO) << "LineSource: " + DeviceIDstr;
					setString<InterfaceModule>("LineSource", DeviceIDstr);

					setString<DeviceModule>("ExposureReadoutOverlap", "True");
					DLOG(INFO) << "AcquisitionFrameRate: " << options.FPS;
					setInteger<DeviceModule>("CycleMinimumPeriod", 1e6 / (2 * options.FPS));

					setString<DeviceModule>("CxpTriggerMessageFormat", "Toggle");

					DLOG(INFO) << "StrobeDuration: " << options.StrobeDuration;
					setInteger<DeviceModule>("StrobeDuration", options.StrobeDuration.Microseconds());

					DLOG(INFO) << "StrobeDelay: " << options.StrobeDelay;
					setInteger<DeviceModule>("StrobeDelay", options.StrobeDelay.Microseconds());

					setString<DeviceModule>("CycleTriggerSource", "Immediate");
				}

				d_width = getInteger<RemoteModule>("Width");
				d_height = getInteger<RemoteModule>("Height");

				DLOG(INFO) << "Width: " << d_width;
				DLOG(INFO) << "Height: " << d_height;

				// Serhii--18.12.2021--
				// Serhii--9.10.2021 setString<RemoteModule>("ExposureMode","Edge_Triggerred_Programmable");

				//setInteger<DeviceModule>("ExposureTime", 6000);
			}
			else
			{
				if (options.SlaveWidth == 0 || options.SlaveHeight == 0)
				{
					throw std::runtime_error("Camera resolution is not specified in DF mode");
				}

				setInteger<RemoteModule>("Width", options.SlaveWidth);
				setInteger<RemoteModule>("Height", options.SlaveHeight);
				d_width = options.SlaveWidth;
				d_height = options.SlaveHeight;
			}

			DLOG(INFO) << "Enable Event";
			enableEvent<NewBufferData>();
			DLOG(INFO) << "Realloc Buffer";
			reallocBuffers(4);  // TODO: Clarify why 4 and whether it should be different in the triggering mode
		}

		void EuresysFrameGrabber::Start()
		{
			DLOG(INFO) << "Starting framegrabber";
			start();
		}

		void EuresysFrameGrabber::Stop()
		{
			DLOG(INFO) << "Stopping framegrabber";
			stop();
			DLOG(INFO) << "/*Framegrabber stopped";
		}

		EuresysFrameGrabber::~EuresysFrameGrabber()
		{
			DLOG(INFO) << "Cleaning FrameGrabber";
		}

		cv::Size EuresysFrameGrabber::Resolution() const
		{
			return cv::Size(d_width, d_height);
		}

		Frame::Ptr EuresysFrameGrabber::NextFrame()
		{
			try
			{
				// processEvent< Euresys::OneOf<Euresys::NewBufferData, Euresys::DataStreamData> >(12000);
				processEvent<Euresys::NewBufferData>(12000);
				// DLOG(INFO) << "NextFrame() - OK!" << std::endl;
				d_eventcount = getInteger<Euresys::DeviceModule>("EventCount[CameraTriggerRisingEdge]");
			}
			catch (const Euresys::gentl_error &err)
			{
				if (err.gc_err == Euresys::gc::GC_ERR_TIMEOUT)
				{
					DLOG(INFO) << "Timeout receiving frame" << std::endl;
				}
				else
				{
					DLOG(INFO) << "GenTL exception: " << err.what() << std::endl;
					throw;
				}
			}

			Frame::Ptr res = d_frame;
			d_frame.reset();
			return res;
		}

		void EuresysFrameGrabber::onNewBufferEvent(const Euresys::NewBufferData &data)
		{
			std::unique_lock<std::mutex> lock(d_mutex);
			d_frame = std::make_shared<EuresysFrame>(*this, data, d_lastFrame, d_toAdd, d_cameraid, d_eventcount);
		}

		EuresysFrame::EuresysFrame(Euresys::EGrabber<Euresys::CallbackOnDemand> &grabber, const Euresys::NewBufferData &data, uint64_t &lastFrame, uint64_t &toAdd, std::string CameraID, uint64_t EventCount)
			: Euresys::ScopedBuffer(grabber, data), d_cameraid(CameraID), d_eventcount(EventCount), d_width(getInfo<size_t>(GenTL::BUFFER_INFO_WIDTH)), d_height(getInfo<size_t>(GenTL::BUFFER_INFO_HEIGHT)), d_timestamp(getInfo<uint64_t>(GenTL::BUFFER_INFO_TIMESTAMP)), d_ID(getInfo<uint64_t>(GenTL::BUFFER_INFO_FRAMEID)), d_mat(d_height, d_width, CV_8U, getInfo<void *>(GenTL::BUFFER_INFO_BASE))
		{
			if (d_ID == 0 && lastFrame != 0)
			{
				toAdd += lastFrame + 1;
			}
			lastFrame = d_ID;
			d_ID += toAdd;
		}

		EuresysFrame::~EuresysFrame() {}

		size_t EuresysFrame::Width() const
		{
			return d_width;
		}
		size_t EuresysFrame::Height() const
		{
			return d_height;
		}

		std::string EuresysFrame::CameraID() const
		{
			return d_cameraid;
		}

		uint64_t EuresysFrame::EventCount() const
		{
			return d_eventcount;
		}

		uint64_t EuresysFrame::Timestamp() const
		{
			return d_timestamp;
		}

		uint64_t EuresysFrame::ID() const
		{
			return d_ID;
		}

		const cv::Mat &EuresysFrame::ToCV()
		{
			return d_mat;
		}

		void *EuresysFrame::Data()
		{
			return getInfo<void *>(GenTL::BUFFER_INFO_BASE);
		}

	} // namespace artemis
} // namespace fort
