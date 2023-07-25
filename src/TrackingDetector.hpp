
//it look like ApriltagDetector.hpp
#pragma once

#include <fort/hermes/FrameReadout.pb.h>

#include "Options.hpp"
#include "utils/Partitions.hpp"

#include <functional>

namespace fort {
namespace artemis {

class TrackingDetector{
public :

	cv::Mat framePrev;
	uint32_t id_frame = 0;

	TrackingDetector(size_t maxParallel, const cv::Size & size, const TrackingOptions & options);

	void Detect(const cv::Mat & mat,
	            //size_t nThreads,
				//bool useCUDA,
	            hermes::FrameReadout & m);

private:

	std::string	trackingmodel;
	std::string	labelfile;
	bool 		useCUDA;
	size_t		trackingthreads;
	
};


} // namespace artemis
} // namespace fort
