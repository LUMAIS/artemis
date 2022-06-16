#pragma once

#include "Options.hpp"
#include  "Task.hpp"

#include <memory>
#include <vector>
#include <boost/asio/io_context.hpp>
#include <boost/asio/signal_set.hpp>


namespace fort {
namespace artemis {

class FrameGrabber;
class AcquisitionTask;
class ProcessFrameTask;
class FullFrameExportTask;

class Application {
public:
	static void Execute(int argc, char ** argv);
	static void applicationrun(const Options & options);


private :
	static bool InterceptCommand(const Options & options);

	static void InitGlobalDependencies();
	static void InitGoogleLogging(char * applicationName,
	                              const GeneralOptions & options);

	Application(const Options & options);

	void Run();

	void SpawnIOContext();
	void SpawnTasks(uint8_t i);
	void JoinTasks();

	typedef boost::asio::executor_work_guard<boost::asio::io_context::executor_type> WorkGuard;

	boost::asio::io_context d_context;
	boost::asio::signal_set d_signals;
	WorkGuard               d_guard;

	//std::shared_ptr<FrameGrabber>        d_grabber;
	//std::shared_ptr<ProcessFrameTask>    d_process;
	//std::shared_ptr<AcquisitionTask>     d_acquisition;

	//for trigger mode

	std::vector<std::shared_ptr<FrameGrabber>>        d_grabber;
	std::vector<std::shared_ptr<ProcessFrameTask>>    d_process;
	std::vector<std::shared_ptr<AcquisitionTask>>     d_acquisition;



	std::vector<std::thread>          d_threads;
	std::thread                       d_ioThread;


};

} // namespace artemis
} // namespace fort
