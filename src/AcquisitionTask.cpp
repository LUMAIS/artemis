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
		std::string triggermode = "none";
		std::string tofile;
		uint renderheight;
		double opt_fps;

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
				bool deviceinit = false;

				triggermode = options.Triggermode;
				renderheight = options.RenderHeight;
				tofile = options.ToFile; // frames will be saved to a file
				opt_fps = options.FPS;

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
							deviceinit = true;

						if (deviceinit)
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
			std::vector<std::tuple<cv::Mat, uint64_t, uint64_t>> fbuf;

			bool mp4conf = false;
			std::string video_filename;
			cv::VideoWriter writer;
			int codec = cv::VideoWriter::fourcc('M', 'P', '4', 'V');
			cv::Size sizeFrame;
			double fps = opt_fps;
			uint8_t nt = 3;

			while (d_quit.load() == false && ptrframe == true)
			{
				Frame::Ptr f = d_grabber->NextFrame();
				if (tofile.length() > 0)
					video_filename = tofile + "_cam_id_" + f->CameraID() + ".mp4";

				if (triggermode != "none")
				{
					cv::Point2f pt, pt1, pt2;
					pt.x = 20.0;
					pt.y = 20.0;

					pt1.x = 10.0;
					pt1.y = 5.0;

					pt2.x = 180.0;
					pt2.y = 25.0;

					if ((f->EventCount() % 3 != std::stoi(f->CameraID()) || f->EventCount() == 0) && triggermode == "sequential")
						continue;

					DLOG(INFO) << "CameraID - " << f->CameraID() << " | EventCount - " << f->EventCount() << std::endl;

					if (tofile.length() > 0)
					{
						if (mp4conf == false)
						{
							if (fbuf.size() < nt)
							{
								fbuf.push_back(std::make_tuple(f->ToCV(), f->Timestamp(), f->EventCount()));
							}
							else
							{
								fps = (double)fbuf.size() / ((std::get<1>(fbuf.at(fbuf.size() - 1)) - std::get<1>(fbuf.at(0))) / 1000000.0);
								writer.open(video_filename, codec, fps, cv::Size(renderheight, renderheight), true);
								mp4conf = true;

								for (int8_t i = 0; i < fbuf.size(); i++)
								{
									cv::Mat frame;
									std::get<0>(fbuf.at(i)).copyTo(frame);
									cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
									frame.convertTo(frame, CV_8UC3);
									if (renderheight > 0)
										cv::resize(frame, frame, cv::Size(renderheight, renderheight), cv::InterpolationFlags::INTER_CUBIC);

									cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 0, 0), cv::FILLED);

									std::string str = "EventCount: " + std::to_string(std::get<2>(fbuf.at(i)));
									cv::putText(frame, // target image
												str,   // text
												pt,	   // top-left position
												1,
												1,
												cv::Scalar(255, 255, 255), // font color
												1);

									writer.write(frame);
								}
							}
						}
						else
						{
							cv::Mat frame;
							f->ToCV().copyTo(frame);
							cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
							frame.convertTo(frame, CV_8UC3);

							if (renderheight > 0)
								cv::resize(frame, frame, cv::Size(renderheight, renderheight), cv::InterpolationFlags::INTER_CUBIC);

							cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 0, 0), cv::FILLED);
							std::string str = "EventCount: " + std::to_string(f->EventCount());
							cv::putText(frame, // target image
										str,   // text
										pt,	   // top-left position
										1,
										1,
										cv::Scalar(255, 255, 255), // font color
										1);

							writer.write(frame);
						}
					}
					
					imshow("Cam_id " + f->CameraID(), f->ToCV());
					cv::waitKey(10);
				}
				else
				{
					if (tofile.length() > 0)
					{
						if (mp4conf == true)
						{
							cv::Mat frame;
							f->ToCV().copyTo(frame);
							cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
							frame.convertTo(frame, CV_8UC3);

							if (renderheight > 0)
								cv::resize(frame, frame, cv::Size(renderheight, renderheight), cv::InterpolationFlags::INTER_CUBIC);

							writer.write(frame);
						}
						else
						{
							writer.open(video_filename, codec, opt_fps, cv::Size(renderheight, renderheight), true);
							mp4conf = true;
						}
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

			if (triggermode != "none" && mp4conf == true)
			{
				writer.release();
			}

			d_grabber->Stop();
			if (d_processFrame)
			{
				d_processFrame->CloseFrameQueue();
			}

			LOG(INFO) << "[AcquisitionTask]:  ended";
		}
	} // namespace artemis
} // namespace fort
