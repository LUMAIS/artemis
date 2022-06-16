#include "Application.hpp"

#include "Options.hpp"

#include <glog/logging.h>
#include <Eigen/Core>

#include <opencv2/core.hpp>

#include <artemis-config.h>

#include "AcquisitionTask.hpp"
#include "ProcessFrameTask.hpp"
#include "FullFrameExportTask.hpp"
#include "VideoOutputTask.hpp"
#include "UserInterfaceTask.hpp"
#include <thread>

namespace fort
{
	namespace artemis
	{
		uint8_t ncams = 1;

		void Application::applicationrun(const Options &options)
		{
			Application application(options);
			application.Run();
		}

		void Application::Execute(int argc, char **argv)
		{
			auto options = Options::Parse(argc, argv, true);
			if (InterceptCommand(options) == true)
			{
				return;
			}

			InitGoogleLogging(argv[0], options.General);
			InitGlobalDependencies();

			Application application(options);

			application.Run();

			/*
			std::thread func_thread(applicationrun,options);
			std::thread func_thread2(applicationrun,options);
			if (func_thread.joinable())
				func_thread.join();

			if (func_thread2.joinable())
				func_thread2.join();
			*/
		};

		bool Application::InterceptCommand(const Options &options)
		{
			if (options.General.PrintVersion == true)
			{
				std::cout << "artemis v" << ARTEMIS_VERSION << std::endl;
				return true;
			}

			if (options.General.PrintResolution == true)
			{
				auto resolution = AcquisitionTask::LoadFrameGrabber(options.General.StubImagePaths, options.General.inputVideoPath,
																	options.Camera)
									  ->Resolution();
				std::cout << resolution.width << " " << resolution.height << std::endl;
				return true;
			}

			return false;
		}

		void Application::InitGoogleLogging(char *applicationName,
											const GeneralOptions &options)
		{
			if (options.LogDir.empty() == false)
			{
				// likely runned by leto and we are saving data to dedicated
				// files. So we do not need as much log to stderr, and no
				// color.
				FLAGS_log_dir = options.LogDir.c_str();
				FLAGS_stderrthreshold = 3;
				FLAGS_colorlogtostderr = false; // maybe we should need less log
			}
			else
			{
				// we output all logs, and likely we are not beeing runned by
				// leto, so we output everything with colors.
				FLAGS_stderrthreshold = 0;
				FLAGS_colorlogtostderr = true;
			}
			::google::InitGoogleLogging(applicationName);
			::google::InstallFailureSignalHandler();
		}

		void Application::InitGlobalDependencies()
		{
			// Needed as we will do some parallelized access to Eigen ??
			Eigen::initParallel();
			// reduce the number of threads for OpenCV to allow some room for
			// other task (Display & IO)
			auto numThreads = cv::getNumThreads();
			if (numThreads == 2)
			{
				numThreads = 1;
			}
			else if (numThreads > 2)
			{
				numThreads -= 2;
			}
			cv::setNumThreads(numThreads);
		}

		Application::Application(const Options &options) : d_signals(d_context, SIGINT), d_guard(d_context.get_executor())
		{
			if (options.Camera.Triggermode)
				ncams = 3;
			
			for (uint8_t i = 0; i < ncams; i++)
			{
				if (options.Camera.Triggermode)
				{
					d_grabber.push_back(AcquisitionTask::LoadFrameGrabber(options.General.StubImagePaths, options.General.inputVideoPath,
																		  options.Camera, std::to_string(i)));
				}
				else
				{
					d_grabber.push_back(AcquisitionTask::LoadFrameGrabber(options.General.StubImagePaths, options.General.inputVideoPath,
																		  options.Camera));
				}

				d_process.push_back(std::make_shared<ProcessFrameTask>(options,
																	   d_context,
																	   d_grabber.at(i)->Resolution()));
			}

			for (uint8_t i = 0; i < ncams; i++)
				d_acquisition.push_back(std::make_shared<AcquisitionTask>(d_grabber.at(i), d_process.at(i)));
		}

		void Application::SpawnTasks(uint8_t i)
		{
			if (d_process.at(i)->FullFrameExportTask())
			{
				d_threads.push_back(Task::Spawn(*d_process.at(i)->FullFrameExportTask(), 20));
			}

			if (d_process.at(i)->VideoOutputTask())
			{
				d_threads.push_back(Task::Spawn(*d_process.at(i)->VideoOutputTask(), 0));
			}

			if (d_process.at(i)->UserInterfaceTask())
			{
				d_threads.push_back(Task::Spawn(*d_process.at(i)->UserInterfaceTask(), 1));
			}

			d_threads.push_back(Task::Spawn(*d_process.at(i), 0));
			d_threads.push_back(Task::Spawn(*d_acquisition.at(i), 0));
		}

		void Application::JoinTasks()
		{
			// we join all subtask, but leave IO task alive
			for (auto &thread : d_threads)
			{
				thread.join();
			}
			// now, nobody will post IO operation. we can reset safely.
			d_guard.reset();
			// we join the IO thread
			d_ioThread.join();
		}

		void Application::SpawnIOContext()
		{
			// putting a wait on SIGINT will ensure that the context remains
			// active throughout execution.
			d_signals.async_wait([this](const boost::system::error_code &,
										int)
								 {
									 LOG(INFO) << "Terminating (SIGINT)";

									 for (uint8_t i = 0; i < ncams; i++)
									 	d_acquisition.at(i)->Stop(); });
			// starts the context in a single threads, and remind to join it
			// once we got the SIGINT
			d_ioThread = std::thread([this]()
									 {
		                         LOG(INFO) << "[IOTask]: started";
		                         d_context.run();
		                         LOG(INFO) << "[IOTask]: ended"; });
		}

		void Application::Run()
		{
			SpawnIOContext();

			for (uint8_t i = 0; i < ncams; i++)
				SpawnTasks(i);

			JoinTasks();
		}

	} // namespace artemis
} // namespace fort
