#include "ProcessQueueExecuter.h"

ProcessQueueExecuter::ProcessQueueExecuter(asio::io_service & service, size_t workers)
	: d_service(service)
	, d_maxWorkers(workers)
	, d_nbActiveWorkers(0) {
}
ProcessQueueExecuter::~ProcessQueueExecuter() {}


void ProcessQueueExecuter::Start(ProcessQueue & queue,const Frame::Ptr & frame) {
	std::lock_guard<std::mutex> lock(d_mutex);
	if (!IsDoneUnsafe()) {
		throw std::runtime_error("Should be done");
	}

	d_current = queue.begin();
	d_end = queue.end();

	d_frame = frame;
	d_results.resize(queue.size()+1);
	d_currentResult = d_results.begin();

	SpawnCurrent();
}

bool ProcessQueueExecuter::IsDone() {
	std::lock_guard<std::mutex> lock(d_mutex);
	return IsDoneUnsafe();
}

bool ProcessQueueExecuter::IsDoneUnsafe() {
	if ( d_nbActiveWorkers > 0 ) {
		return false;
	}

	if (d_current == d_end ) {
		return true;
	}
}


void ProcessQueueExecuter::SpawnCurrent() {
	auto jobs = (*d_current)->Prepare(d_maxWorkers,d_frame->ToCV().size());
	{
		std::lock_guard<std::mutex> lock(d_mutex);
		d_nbActiveWorkers += jobs.size();
	}

	for(auto & j : jobs) {
		d_service.post([this,j](){
				j(d_frame,*d_currentResult,d_message,*(d_currentResult+1));
				{
					std::lock_guard<std::mutex> lock(d_mutex);
					if ( --d_nbActiveWorkers != 0 ) {
						//still unfinished work
						return;
					}
					if (++d_current == d_end){
						// no more job to do
						return;
					}
					++d_currentResult;
				}
				SpawnCurrent();
			});
	}

}