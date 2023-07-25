#include "Application.hpp"

#include "Options.hpp"

#include <glog/logging.h>
#include <Eigen/Core>

#include <opencv2/core.hpp>

#include <artemis-config.h>

#include "AcquisitionTask.hpp"
#include "ProcessFrameTask.hpp"
#include "FullFrameExportTask.hpp"
#include "VideoStdoutTask.hpp"
#include "UserInterfaceTask.hpp"
#include <thread>

namespace fort
{
	namespace artemis
	{
		void Application::applicationrun(const Options &options, bool UITask)
		{
			Application application(options);
			application.Run(UITask);
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

			if (options.Camera.Triggermode != "none" && options.Camera.cameraID.empty())
			{
				auto options_0 = options;
				options_0.Camera.cameraID = "0";
				options_0.Camera.toDisplayFrame = true;
				auto options_1 = options;
				options_1.Camera.cameraID = "1";
				options_1.Camera.toDisplayFrame = false;
				auto options_2 = options;
				options_2.Camera.cameraID = "2";
				options_2.Camera.toDisplayFrame = false;

				std::thread func_thread_0(applicationrun, options_0, true);
				std::thread func_thread_1(applicationrun, options_1, false);
				std::thread func_thread_2(applicationrun, options_2, false);

				if (func_thread_0.joinable())
					func_thread_0.join();

				if (func_thread_1.joinable())
					func_thread_1.join();

				if (func_thread_2.joinable())
					func_thread_2.join();
			}
			else
			{
				Application application(options);
				application.Run(true);
			}
		};

		bool Application::InterceptCommand(const Options &options)
		{
			if (options.General.PrintVersion == true)
			{
				std::cout << "artemis v" << ARTEMIS_VERSION
// GIT_SRC_VERSION (a custom macro definition) might be defined by CMAKE or another build tool
#ifdef GIT_SRC_VERSION
					<< " (" << GIT_SRC_VERSION << ")"
#endif // GIT_SRC_VERSION
				<< std::endl;
				return true;
			}

			if (options.General.PrintResolution == true)
			{
				auto resolution = AcquisitionTask::LoadFrameGrabber(options.General.StubImagePaths, options.General.inputVideoPath,
																	options.Camera, options.VideoOutput)
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
			// Fetch frame grabbing interface (per each camera)
			std::shared_ptr<FrameGrabber>  grabber = AcquisitionTask::LoadFrameGrabber(options.General.StubImagePaths, options.General.inputVideoPath, options.Camera, options.VideoOutput);

			// Define processing tasks per each grabber interface
			d_tasks.push_back(std::make_shared<AcquisitionTask>(grabber,
				std::make_shared<ProcessFrameTask>(options, d_context, grabber->Resolution()))
			);
		}

		void Application::SpawnTasks(bool UITask)
		{
			for(auto& task: d_tasks)
			{
				if (task->process()->FullFrameExportTask())
				{
					d_threads.push_back(Task::Spawn(*task->process()->FullFrameExportTask(),
	#ifdef HAVE_OPENCV_CUDACODEC
						1
	#else
						20
	#endif // HAVE_OPENCV_CUDACODEC
					));
				}

				if (task->process()->VideoStdoutTask())
				{
					d_threads.push_back(Task::Spawn(*task->process()->VideoStdoutTask(), 0));
				}

				if (UITask)
				{
					if (task->process()->UserInterfaceTask())
					{
						d_threads.push_back(Task::Spawn(*task->process()->UserInterfaceTask(), 1));
					}
				}
				
				// Schedule frame processing and acquisition (grabbing) tasks
				d_threads.push_back(Task::Spawn(*task->process(), 0));
				d_threads.push_back(Task::Spawn(*task, 0));
			}
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
			d_signals.async_wait([this](const boost::system::error_code &, int)
								{
									 LOG(INFO) << "Terminating (SIGINT)";

									 for (auto& task: d_tasks)
									 	task->Stop();
								});
			// starts the context in a single threads, and remind to join it
			// once we got the SIGINT
			d_ioThread = std::thread([this]()
								{
									LOG(INFO) << "[IOTask]: started";
									d_context.run();
									LOG(INFO) << "[IOTask]: ended";
								});
		}

		void Application::Run(bool UITask)
		{
			SpawnIOContext();
			SpawnTasks(UITask);
			JoinTasks();
		}

	} // namespace artemis
} // namespace fort
