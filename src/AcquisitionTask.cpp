#include "AcquisitionTask.hpp"

#include <filesystem>
#include <glog/logging.h>
#include <artemis-config.h>
#ifndef FORCE_STUB_FRAMEGRABBER_ONLY
#include "EuresysFrameGrabber.hpp"
#endif // FORCE_STUB_FRAMEGRABBER_ONLY
#include "StubFrameGrabber.hpp"
#include "StubVideoGrabber.hpp"

#include "ProcessFrameTask.hpp"

namespace fort
{
	namespace artemis
	{
		namespace fs = std::filesystem;

		std::string triggermode = "none";
		std::string tofile;
		uint vidH = 0;
		double opt_fps = 0;

		FrameGrabber::Ptr AcquisitionTask::LoadFrameGrabber(const std::vector<std::string> &stubImagePaths, std::string inputVideoPath,
															const CameraOptions &options, const VideoOutputOptions & vidOpts, std::string cameraID)
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
				opt_fps = options.FPS;
				vidH = vidOpts.Height;
				tofile = vidOpts.ToFile; // frames will be saved to a file

				if (!tofile.empty()) {
					// Ensure that the output directory exists
					// Fetch base dir
					size_t pos = tofile.find_last_of("\\/");
					std::string dir = std::string::npos == pos ? "" : tofile.substr(0, pos);
					const fs::path  odp = dir;
					std::error_code  ec;
					if (!odp.empty() && !fs::exists(odp, ec)) {
						if(ec || !fs::create_directories(odp, ec))
							throw std::runtime_error("The output directory can't be created (" + std::to_string(ec.value()) + ": " + ec.message() + "): " + odp.string() + "\n");
					}
				}

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

		ProcessFrameTaskPtr& AcquisitionTask::process()
		{
			return d_processFrame;
		}

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
#ifdef HAVE_OPENCV_CUDACODEC
#else
#endif // HAVE_OPENCV_CUDACODEC
			cv::VideoWriter writer;

			bool mp4conf = false;
			std::string video_filename;
			int codec = cv::VideoWriter::fourcc('M', 'P', '4', 'V');
			cv::Size sizeFrame;
			double fps = opt_fps;
			uint8_t nt = 3;  // The number of cashed initial frames used to evaluate FPS

			while (d_quit.load() == false && ptrframe == true)
			{
				Frame::Ptr f = d_grabber->NextFrame();
				if (!tofile.empty())
					video_filename = tofile + "_CamId-" + f->CameraID() + ".mp4";

				const cv::Size  sz = vidH > 0
					? cv::Size(round((double)vidH / f->Height() * f->Width()), vidH)
					: cv::Size(f->Width(), f->Height());

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

					if (!tofile.empty())
					{
						if (mp4conf == false)
						{
							if (fbuf.size() < nt)
							{
								fbuf.push_back(std::make_tuple(f->ToCV(), f->Timestamp(), f->EventCount()));
							}
							else
							{
								fps = (double)fbuf.size() / ((std::get<1>(fbuf.back()) - std::get<1>(fbuf.front())) / 1E6);
								writer.open(video_filename, codec, fps, sz, f->ToCV().channels() > 1);
								mp4conf = true;

								for (int8_t i = 0; i < fbuf.size(); i++)
								{
									cv::Mat frame;
									std::get<0>(fbuf.at(i)).copyTo(frame);
									if(frame.channels() > 1) {
										cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
										assert(frame.type() == CV_8UC3 && "Unexpexted frame type");
									}
									if (vidH != frame.rows)
										cv::resize(frame, frame, sz, cv::InterpolationFlags::INTER_CUBIC);

									cv::rectangle(frame, pt1, pt2, cv::Scalar::all(0), cv::FILLED);

									std::string str = "EventCount: " + std::to_string(std::get<2>(fbuf.at(i)));
									cv::putText(frame, // target image
												str,   // text
												pt,	   // top-left position
												1,
												1,
												cv::Scalar::all(255), // font color
												1);

									writer.write(frame);
								}
							}
						}
						else
						{
							cv::Mat frame;
							f->ToCV().copyTo(frame);
							if(frame.channels() > 1) {
								cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
								assert(frame.type() == CV_8UC3 && "Unexpexted frame type");
							}

							if (vidH != frame.rows)
								cv::resize(frame, frame, sz, cv::InterpolationFlags::INTER_CUBIC);

							cv::rectangle(frame, pt1, pt2, cv::Scalar::all(0), cv::FILLED);
							
							std::string str = "EventCount: " + std::to_string(f->EventCount());
							cv::putText(frame, // target image
										str,   // text
										pt,	   // top-left position
										1,
										1,
										cv::Scalar::all(255), // font color
										1);

							writer.write(frame);
						}
					}
					
					imshow("Cam_id " + f->CameraID(), f->ToCV());
					cv::waitKey(10);
				}
				else
				{
					if (!tofile.empty())
					{
						if (mp4conf == true)
						{
							cv::Mat frame;
							f->ToCV().copyTo(frame);
							if(frame.channels() > 1) {
								cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
								assert(frame.type() == CV_8UC3 && "Unexpexted frame type");
							}

							if (vidH != frame.rows)
								cv::resize(frame, frame, sz, cv::InterpolationFlags::INTER_CUBIC);

							writer.write(frame);
						}
						else
						{
							writer.open(video_filename, codec, opt_fps, sz, f->ToCV().channels() > 1);
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
