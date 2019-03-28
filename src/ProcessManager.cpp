#include "ProcessManager.h"

#include "EventManager.h"
#include "ProcessDefinition.h"

ProcessManager::Worker::Worker()
	: d_quit(false)
	, d_workThread(&ProcessManager::Worker::Loop,this){
}

ProcessManager::Worker::~Worker() {
	d_quit.store(true);
	{
		std::lock_guard<std::mutex> lock(d_mutex);
		d_signal.notify_all();
	}
	d_workThread.join();
}


void ProcessManager::Worker::StartJob(WaitGroup & wg, const EventManager::Ptr & eventManager, const Job & j) {
	std::lock_guard<std::mutex> lock(d_mutex);

	d_job = std::make_shared<Job>([&wg,j,eventManager]{
			j();
			wg.Done();
			eventManager->Signal(Event::PROCESS_NEED_REFRESH);
		});

	wg.Add(1);
	d_signal.notify_all();
}

void ProcessManager::Worker::Loop() {
	while(d_quit.load() == false) {
		std::shared_ptr<Job> toDo;
		{
			std::unique_lock<std::mutex> lock(d_mutex);
			d_signal.wait(lock,  [this,&toDo]() -> bool {
					if ( d_quit.load() == true ) {
						return true;
					}
					if (!d_job ) {
						return false;
					}
					toDo = d_job;
					d_job.reset();
					return true;
				});
			if (d_quit.load() == true ) {
				return;
			}
		}
		(*toDo)();
	}

}

ProcessManager::ProcessManager(const EventManagerPtr & eventManager,
                               size_t nbWorkers)
	: d_eventManager(eventManager) {
	if (nbWorkers == 0 ) {
		throw std::invalid_argument("ProcessManager: Need at least 1 worker");
	}
	if ( !eventManager ) {
		throw std::invalid_argument("ProcessManager: empty EventManager reference");
	}

	d_workers.reserve(nbWorkers);
	for ( size_t i = 0; i < nbWorkers; ++i ) {
		d_workers.push_back(std::unique_ptr<Worker>(new Worker()));
	}
}


void ProcessManager::Start(const std::vector<std::function < void()> > & jobs) {
	if (jobs.size() > d_workers.size() ) {
		throw std::invalid_argument("Too many jobs");
	}

	for(size_t i = 0; i < jobs.size(); ++i) {
		d_workers[i]->StartJob(d_waitGroup,d_eventManager,jobs[i]);
	}

}


bool ProcessManager::IsDone() {
	return d_waitGroup.IsDone();
}


void ProcessManager::Wait() {
	d_waitGroup.Wait();
}
