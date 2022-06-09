#include "AcquisitionTask.hpp"

#include <artemis-config.h>
#ifndef FORCE_STUB_FRAMEGRABBER_ONLY
#include "EuresysFrameGrabber.hpp"
#endif // FORCE_STUB_FRAMEGRABBER_ONLY
#include "StubFrameGrabber.hpp"
#include "StubVideoGrabber.hpp"

#include "ProcessFrameTask.hpp"

#include <glog/logging.h>

namespace fort
{
	namespace artemis
	{
		bool triggermode;
		FrameGrabber::Ptr AcquisitionTask::LoadFrameGrabber(const std::vector<std::string> &stubImagePaths, std::string inputVideoPath,
															const CameraOptions &options, std::string cameraID)
		{
#ifndef FORCE_STUB_FRAMEGRABBER_ONLY
			if (stubImagePaths.empty() && inputVideoPath.length() == 0)
			{
				static Euresys::EGenTL egentl;

				using namespace Euresys;

				int interfaceID = 0;
				int deviceID = 0;

				if (options.Triggermode)
					triggermode = true;
				else
					triggermode = false;

				//--Serhii--8.10.2021
				gc::TL_HANDLE tl = egentl.tlOpen();
				uint32_t numInterfaces = egentl.tlGetNumInterfaces(tl);
				LOG(INFO) << "[LoadFrameGrabber]:  numInterfaces - " << numInterfaces;

				for (uint32_t interfaceIndex = 0; interfaceIndex < numInterfaces; interfaceIndex++)
				{
					std::string interfaceID = egentl.tlGetInterfaceID(tl, interfaceIndex);
					gc::IF_HANDLE interfaceHandle = egentl.tlOpenInterface(tl, interfaceID);
					uint32_t numDevice = egentl.ifGetNumDevices(interfaceHandle);
					LOG(INFO) << "[LoadFrameGrabber]:  numDevice - " << numDevice;

					for (uint32_t deviceIndex = 0; deviceIndex < numDevice; deviceIndex++)
					{
						std::string deviceIDstring = egentl.ifGetDeviceID(interfaceHandle, deviceIndex);
						gc::DEV_HANDLE deviceHandle = egentl.ifOpenDevice(interfaceHandle, deviceIDstring);

						LOG(INFO) << "[LoadFrameGrabber]:  deviceIndex - " << deviceIndex;

						try
						{
							if (egentl.devGetPort(deviceHandle))
							{
								// grabbers.push_back(new MyGrabber(genTL, interfaceIndex, deviceIndex, interfaceID, deviceID));
								// LOG(INFO) << "[LoadFrameGrabber]: Camera connected OK "<<interfaceID<<" <"<<deviceID<<">"<<std::endl;
							}
						}
						catch (const Euresys::gentl_error &)
						{
							LOG(INFO) << "[LoadFrameGrabber]: no camera connected on " << interfaceID << " <" << deviceID << ">" << std::endl;
						}

						if (std::stoi(options.cameraID) == deviceIndex)
						{
							interfaceID = interfaceIndex;
							deviceID = deviceIndex;
							interfaceIndex = numInterfaces;
							deviceIndex = numDevice;
							LOG(INFO) << "[LoadFrameGrabber]: Camera selected " << deviceID << std::endl;

							break;
						}
					}
				}
				//--Serhii--8.10.2021
				return std::make_shared<EuresysFrameGrabber>(egentl, options, interfaceID, deviceID);
			}
			else
			{
				if (!stubImagePaths.empty())
				{
					// return std::make_shared<StubFrameGrabber>(stubImagePaths,options.FPS);
					return std::make_shared<StubFrameGrabber>(stubImagePaths, options);
				}
				else
				{
					// return std::make_shared<StubVideoGrabber>(inputVideoPath,options.FPS);
					return std::make_shared<StubVideoGrabber>(inputVideoPath, options);
				}
			}
#else
			return std::make_shared<StubFrameGrabber>(stubImagePaths, options.FPS);
#endif
		}

		AcquisitionTask::AcquisitionTask(const FrameGrabber::Ptr &grabber,
										 const ProcessFrameTaskPtr &process)
			: d_grabber(grabber), d_processFrame(process)
		{
			d_quit.store(false);
		}

		AcquisitionTask::~AcquisitionTask() {}

		void AcquisitionTask::Stop()
		{
			d_quit.store(true);
		}

		void AcquisitionTask::Run()
		{
			LOG(INFO) << "[AcquisitionTask]:  started";

			d_grabber->Start();
			bool ptrframe = true;

			std::vector<std::pair<cv::Mat, uint64_t>> fbuf;
			bool mp4conf = false;
			std::string video_filename;
			cv::VideoWriter writer;
			int codec = cv::VideoWriter::fourcc('M', 'P', '4', 'V');
			cv::Size sizeFrame;
			double fps = 1.0;
			uint8_t nt = 9;

			while (d_quit.load() == false && ptrframe == true)
			{
				Frame::Ptr f = d_grabber->NextFrame();

				if (triggermode)
				{
					DLOG(INFO) << "EventCount - " << f->EventCount() << std::endl;

					if (f->EventCount() % 3 != std::stoi(f->CameraID()))
						continue;

					LOG(INFO) << "[AcquisitionTask]:  f->Timestamp() - " << f->Timestamp() / 100000;

					if (mp4conf == false)
					{
						if (fbuf.size() < nt)
						{

							fbuf.push_back(std::make_pair(f->ToCV(), f->Timestamp()));
						}
						else
						{
							fps = (double)fbuf.size() / ((fbuf.at(fbuf.size() - 1).second - fbuf.at(0).second) / 1000000.0);
							LOG(INFO) << "[AcquisitionTask]:  fps - " << fps;

							video_filename = "cam_id_" + f->CameraID() + ".mp4";
							writer.open(video_filename, codec, fps, cv::Size(1000, 1000), true);
							mp4conf = true;

							for (int8_t i = 0; i < fbuf.size(); i++)
							{
								cv::Mat frame;
								fbuf.at(i).first.copyTo(frame);
								cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
								frame.convertTo(frame, CV_8UC3);
								cv::resize(frame, frame, cv::Size(1000, 1000), cv::InterpolationFlags::INTER_CUBIC);
								writer.write(frame);
							}
						}
					}

					if (mp4conf == true)
					{
						cv::Mat frame;
						f->ToCV().copyTo(frame);
						cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
						frame.convertTo(frame, CV_8UC3);
						cv::resize(frame, frame, cv::Size(1000, 1000), cv::InterpolationFlags::INTER_CUBIC);
						writer.write(frame);
					}
				}

				if (!f)
					ptrframe = false;

				if (d_processFrame)
				{
					d_processFrame->QueueFrame(f);
				}
			}

			LOG(INFO) << "[AcquisitionTask]:  Tear Down";

			if (triggermode == true && mp4conf == true)
			{
				writer.release();
			}

			d_grabber->Stop();
			if (d_processFrame)
			{
				d_processFrame->CloseFrameQueue();
			}

			LOG(INFO) << "[AcquisitionTask]:  ended";
			sleep(5);
			raise(SIGINT);
		}

	} // namespace artemis
} // namespace fort
