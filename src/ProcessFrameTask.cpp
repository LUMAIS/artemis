#include "ProcessFrameTask.hpp"

#include <filesystem>

#include <tbb/parallel_for.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <artemis-config.h>

#include "Connection.hpp"
#include "ApriltagDetector.hpp"
#include "TrophallaxisDetector.hpp"
#include "TrackingDetector.hpp"
#include "FullFrameExportTask.hpp"
#include "VideoStdoutTask.hpp"
#include "UserInterfaceTask.hpp"

#include <glog/logging.h>

#include "Utils.hpp"

namespace fort
{
	namespace artemis
	{
		namespace fs = std::filesystem;

		ProcessFrameTask::ProcessFrameTask(const Options &options,
										   boost::asio::io_context &context,
										   const cv::Size &inputResolution)
			: d_options(options.Process), d_maximumThreads(cv::getNumThreads())
		{
			d_actualThreads = d_maximumThreads;
			d_workingResolution = options.VideoOutput.WorkingResolution(inputResolution);

			SetUpDetection(inputResolution, options.Apriltag);
			SetUpDetectionTrophallaxis(inputResolution, options.Trophallaxis);
			SetUpDetectionTracking(inputResolution, options.Tracking);
			if(!options.Display.NoGUI)
				SetUpUserInterface(d_workingResolution, inputResolution, options);
			SetUpVideoOutputTask(options.VideoOutput, context, options.General.LegacyMode);
			SetUpCataloguing(options.Process);
			SetUpTracing(options.Process);
			SetUpPoolObjects();
			SetUpConnection(options.Network, context);

			std::string ids, prefix;
			for (const auto &id : options.Process.FrameID)
			{
				ids += prefix + std::to_string(id);
				prefix = ",";
			}

			d_todisplayframe = options.Camera.toDisplayFrame;
			LOG(INFO) << "Processing IDs: " << ids;
			LOG(INFO) << "Processing Stride: " << options.Process.FrameStride;
		}

		VideoOutputTaskPtr ProcessFrameTask::VideoStdoutTask() const
		{
			return d_videoOutput;
		}

		UserInterfaceTaskPtr ProcessFrameTask::UserInterfaceTask() const
		{
			return d_userInterface;
		}

		FullFrameExportTaskPtr ProcessFrameTask::FullFrameExportTask() const
		{
			return d_fullFrameExport;
		}

		void ProcessFrameTask::SetUpVideoOutputTask(const VideoOutputOptions &options,
													boost::asio::io_context &context,
													bool legacyMode)
		{
			if (options.ToStdout == false)
			{
				return;
			}
			d_videoOutput = std::make_shared<artemis::VideoStdoutTask>(options, context, legacyMode);
		}

		void ProcessFrameTask::SetUpDetection(const cv::Size &inputResolution,
											  const ApriltagOptions &options)
		{
			if (options.Family == tags::Family::Undefined)
			{
				return;
			}
			d_detector = std::make_unique<ApriltagDetector>(d_maximumThreads,
															inputResolution,
															options);
		}

		void ProcessFrameTask::SetUpDetectionTrophallaxis(const cv::Size &inputResolution,
														  const TrophallaxisOptions &options)
		{
			if (options.trophallaxismodel.length() == 0)
			{
				return;
			}
			d_detectorTroph = std::make_unique<TrophallaxisDetector>(d_maximumThreads,
																	 inputResolution,
																	 options);
		}

		void ProcessFrameTask::SetUpDetectionTracking(const cv::Size &inputResolution,
													  const TrackingOptions &options)
		{
			if (options.trackingmodel.length() == 0)
			{
				return;
			}
			d_detectorTrack = std::make_unique<TrackingDetector>(d_maximumThreads,
																 inputResolution,
																 options);
		}

		void ProcessFrameTask::SetUpCataloguing(const ProcessOptions &options)
		{
			if (options.NewAntOutputDir.empty())
			{
				return;
			}

			d_nextAntCatalog = Time::Now();
			d_nextFrameExport = d_nextAntCatalog.Add(10 * Duration::Second);

			d_fullFrameExport = std::make_shared<artemis::FullFrameExportTask>(options.NewAntOutputDir);
		}

		void ProcessFrameTask::SetUpTracing(const ProcessOptions &options)
		{
			if (options.AntTraceFile.empty())
				return;
			// Fetch base dir
 			size_t pos = options.AntTraceFile.find_last_of("\\/");
			std::string dir = std::string::npos == pos ? "" : options.AntTraceFile.substr(0, pos);
			const fs::path  odp = dir;
			std::error_code  ec;
			if (!odp.empty() && !fs::exists(odp, ec)) {
				if(ec || !fs::create_directories(odp, ec))
					throw std::runtime_error("The output directory can't be created (" + std::to_string(ec.value()) + ": " + ec.message() + "): " + odp.string() + "\n");
			}

			d_ftrace = std::fstream(options.AntTraceFile, std::ios_base::out);
			d_ftrace << "# FrameId TagId ObjCenterX ObjCenterY\n" << std::fixed << std::setprecision(2);
		}

		void ProcessFrameTask::SetUpConnection(const NetworkOptions &options,
											   boost::asio::io_context &context)
		{
			if (options.Host.empty())
			{
				return;
			}
			d_connection = Connection::Create(context, options.Host, options.Port, 5 * Duration::Second);
		}

		void ProcessFrameTask::SetUpUserInterface(const cv::Size &workingResolution,
												  const cv::Size &fullResolution,
												  const Options &options)
		{
			d_userInterface = std::make_shared<artemis::UserInterfaceTask>(workingResolution,
																		   fullResolution,
																		   options);

			d_wantedROI = d_userInterface->DefaultROI();
		}

		void ProcessFrameTask::SetUpPoolObjects()
		{
			d_grayImagePool.Reserve(GrayscaleImagePerCycle() * ARTEMIS_FRAME_QUEUE_CAPACITY,
									d_workingResolution.height,
									d_workingResolution.width,
									CV_8UC1);

			d_rgbImagePool.Reserve(RGBImagePerCycle() * ARTEMIS_FRAME_QUEUE_CAPACITY,
								   d_workingResolution.height,
								   d_workingResolution.width,
								   CV_8UC3);
		}

		ProcessFrameTask::~ProcessFrameTask() {}

		void ProcessFrameTask::TearDown()
		{
			if (d_userInterface)
			{
				d_userInterface->CloseQueue();
			}

			if (d_fullFrameExport)
			{
				d_fullFrameExport->CloseQueue();
			}

			if (d_videoOutput)
			{
				d_videoOutput->CloseQueue();
			}
		}

		void ProcessFrameTask::Run()
		{
			LOG(INFO) << "[ProcessFrameTask]: Started";
			Frame::Ptr frame;
			d_frameDropped = 0;
			d_frameProcessed = 0;
			d_start = Time::Now();
			for (;;)
			{
				d_frameQueue.pop(frame);

				if (!frame)
				{
					break;
				}

				//--Serhii--12.10.2021--
				// cv::Mat testframe = frame->ToCV();

				// LOG(INFO) << "[ProcessFrameTask]: frame->cols - " <<frame->Width();

				// cv::Mat resized_frame;

				// resize(testframe, frame->ToCV(), cv::Size(frame->Width(), frame->Height()), cv::INTER_LINEAR);

				// resize(testframe, frame->ToCV(), cv::Size(790, 543), cv::INTER_LINEAR);
				// imgpath = "img_"+interfaceId+"_"+deviceId+".bmp";
				// cv::imwrite(imgpath, resized_frame);
				// cv::imwrite("test_b.bmp", testframe);
				// cv::imshow("Test Frame", resized_frame);
				// cv::waitKey(1000);
				//--Serhii--12.10.2021--

				if (d_fullFrameExport && d_fullFrameExport->IsFree())
				{
					d_actualThreads = d_maximumThreads;
					cv::setNumThreads(d_actualThreads);
				}

				ProcessFrameMandatory(frame);

				if (d_frameQueue.size() > 0)
				{
					if (ShouldProcess(frame->ID()) == true)
					{
						DropFrame(frame);
					}
					continue;
				}

				ProcessFrame(frame);
			}
			LOG(INFO) << "[ProcessFrameTask]: Tear Down";
			TearDown();
			LOG(INFO) << "[ProcessFrameTask]: Ended";
		}

		void ProcessFrameTask::ProcessFrameMandatory(const Frame::Ptr &frame)
		{
			if (!d_videoOutput && !d_userInterface)
			{
				return;
			}
			d_downscaled = d_grayImagePool.Get();
			cv::resize(frame->ToCV(), *d_downscaled, d_workingResolution, 0, 0, cv::INTER_NEAREST);

			if (d_videoOutput)
			{
				auto converted = d_rgbImagePool.Get();
				cv::cvtColor(*d_downscaled, *converted, cv::COLOR_GRAY2RGB);
				d_videoOutput->QueueFrame(converted, frame->Time(), frame->ID());
			}

			// user interface communication will happen after in DisplayFrame.
		}

		void ProcessFrameTask::DropFrame(const Frame::Ptr &frame)
		{
			++d_frameDropped;
			LOG(WARNING) << "Frame dropped due to over-processing. Total dropped: "
						 << d_frameDropped
						 << " ("
						 << 100.0 * double(d_frameDropped) / double(d_frameDropped + d_frameProcessed)
						 << "%)";

			if (!d_connection)
			{
				return;
			}

			auto m = PrepareMessage(frame);
			m->set_error(hermes::FrameReadout::PROCESS_OVERFLOW);

			Connection::PostMessage(d_connection, *m);
		}

		void ProcessFrameTask::ProcessFrame(const Frame::Ptr &frame)
		{
			auto m = PrepareMessage(frame);

			if (ShouldProcess(frame->ID()) == true)
			{
				Detect(frame, *m);
				if (d_connection)
				{
					Connection::PostMessage(d_connection, *m);
				}

				CatalogAnt(frame, *m);
				TraceAnts(*m);

				ExportFullFrame(frame);

				++d_frameProcessed;
			}

			if (d_todisplayframe)
				DisplayFrame(frame, m);
		}

		std::shared_ptr<hermes::FrameReadout> ProcessFrameTask::PrepareMessage(const Frame::Ptr &frame)
		{
			auto m = d_messagePool.Get();
			m->Clear();
			m->set_timestamp(frame->Timestamp());
			m->set_frameid(frame->ID());
			frame->Time().ToTimestamp(m->mutable_time());
			m->set_producer_uuid(d_options.UUID);
			m->set_width(frame->Width());
			m->set_height(frame->Height());
			return m;
		}

		bool ProcessFrameTask::ShouldProcess(uint64_t ID)
		{
			if (d_options.FrameStride <= 1)
			{
				return true;
			}
			return d_options.FrameID.count(ID % d_options.FrameStride) != 0;
		}

		void ProcessFrameTask::QueueFrame(const Frame::Ptr &frame)
		{
			d_frameQueue.push(frame);
		}

		void ProcessFrameTask::CloseFrameQueue()
		{
			d_frameQueue.push({});
		}

		void ProcessFrameTask::Detect(const Frame::Ptr &frame,
									  hermes::FrameReadout &m)
		{
			// Note: all detectors expect grayscale frames
			cv::Mat framebuf = frame->ToCV();
			assert(framebuf.type() == CV_8UC1 && "Grayscale input frames are expected");
			// if(framebuf.channels() == 3) // frame.type() == CV_8UC3
			// 	cv::cvtColor(framebuf, framebuf, cv::COLOR_BGR2GRAY);

			if (d_detector)
			{
				d_detector->Detect(framebuf, d_actualThreads, m);
			}

			if (d_detectorTroph)
			{
				d_detectorTroph->Detect(framebuf, m);
			}

			if (d_detectorTrack)
			{
				d_detectorTrack->Detect(framebuf, m);
			}
		}

		void ProcessFrameTask::ExportFullFrame(const Frame::Ptr &frame)
		{
			if (!d_fullFrameExport || frame->Time().Before(d_nextFrameExport))
			{
				return;
			}

			if (d_fullFrameExport->QueueExport(frame) == false)
			{
				return;
			}
			d_actualThreads = d_maximumThreads - 1;
			cv::setNumThreads(d_actualThreads);
			d_nextFrameExport = frame->Time().Add(d_options.ImageRenewPeriod);
		}

		void ProcessFrameTask::TraceAnts(const hermes::FrameReadout & m)
		{
			if (!d_ftrace)
				return;
			for (const auto &t : m.tags())
				d_ftrace << m.frameid() << ' ' << t.id() << ' ' << t.x() << ' ' << t.y() << std::endl;  //  << m.timestamp()
			// LOG(INFO) << "[ProcessFrameTask]: d_ftrace updated with tags: " << m.tags().size();
		}

		void ProcessFrameTask::CatalogAnt(const Frame::Ptr &frame,
										  const hermes::FrameReadout &m)
		{
			if (d_options.NewAntOutputDir.empty())
			{
				return;
			}

			ResetExportedID(frame->Time());

			auto toExport = FindUnexportedID(m);
			tbb::parallel_for(std::size_t(0),
							  toExport.size(),
							  [&](std::size_t index)
							  {
								  const auto &[tagID, x, y] = toExport[index];
								  ExportROI(frame->ToCV(), frame->ID(), tagID, x, y);
							  });
			for (const auto &[tagID, x, y] : toExport)
			{
				d_exportedID.insert(tagID);
			}
		}

		void ProcessFrameTask::ResetExportedID(const Time &time)
		{
			if (d_nextAntCatalog.Before(time))
			{
				d_nextAntCatalog = time.Add(d_options.ImageRenewPeriod);
				d_exportedID.clear();
			}
		}

		std::vector<std::tuple<uint32_t, double, double>>
		ProcessFrameTask::FindUnexportedID(const hermes::FrameReadout &m)
		{
			std::vector<std::tuple<uint32_t, double, double>> res;
			for (const auto &t : m.tags())
			{
				if (d_exportedID.count(t.id()) != 0)
				{
					continue;
				}
				res.push_back({t.id(), t.x(), t.y()});
				if (res.size() >= d_actualThreads)
				{
					break;
				}
			}
			return res;
		}

		void ProcessFrameTask::ExportROI(const cv::Mat &image,
										 uint64_t frameID,
										 uint32_t tagID,
										 double x,
										 double y)
		{

			std::ostringstream oss;
			oss << d_options.NewAntOutputDir << "/ant_" << tagID << "_" << frameID << ".png";
			cv::imwrite(oss.str(),
						cv::Mat(image,
								GetROICenteredAt({int(x), int(y)},
												 cv::Size(d_options.NewAntROISize,
														  d_options.NewAntROISize),
												 image.size())));
		}

		void ProcessFrameTask::DisplayFrame(const Frame::Ptr frame, const std::shared_ptr<hermes::FrameReadout> &m)
		{
			if(!d_userInterface)
				return;

			d_wantedROI = d_userInterface->UpdateROI(d_wantedROI);
			std::shared_ptr<cv::Mat> zoomed;
			if (d_wantedROI.size() != frame->ToCV().size())
			{
				zoomed = d_grayImagePool.Get();
				cv::Size size = frame->ToCV().size();
				cv::resize(cv::Mat(frame->ToCV(), d_wantedROI), *zoomed, d_workingResolution, 0, 0, cv::INTER_NEAREST);
			}

			UserInterface::FrameToDisplay toDisplay =
				{
					.Full = d_downscaled,
					.Zoomed = zoomed,
					.Message = m,
					.CurrentROI = d_wantedROI,
					.FrameTime = frame->Time(),
					.FPS = CurrentFPS(frame->Time()),
					.FrameProcessed = d_frameProcessed,
					.FrameDropped = d_frameDropped,
					.VideoOutputProcessed = -1UL,
					.VideoOutputDropped = -1UL,
					.CameraID = frame->CameraID(),
					
					// .RenderHeight = RenderHeight,
				};

			if (d_videoOutput != nullptr)
			{
				toDisplay.VideoOutputProcessed = d_videoOutput->FrameProcessed();
				toDisplay.VideoOutputDropped = d_videoOutput->FrameDropped();
			}

			d_userInterface->QueueFrame(toDisplay);
		}

		size_t ProcessFrameTask::RGBImagePerCycle() const
		{
			if (d_videoOutput)
			{
				return 1;
			}
			return 0;
		}

		size_t ProcessFrameTask::GrayscaleImagePerCycle() const
		{
			if (!d_videoOutput && !d_userInterface)
			{
				return 0;
			}
			if (d_userInterface)
			{
				return 2;
			}
			return 1;
		}

		double ProcessFrameTask::CurrentFPS(const Time &time)
		{
			return d_frameProcessed / time.Sub(d_start).Seconds();
		}

	} // namespace artemis
} // namespace fort
