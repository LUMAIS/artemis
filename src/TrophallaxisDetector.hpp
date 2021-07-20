//it look like ApriltagDetector.hpp
#pragma once

#include <fort/hermes/FrameReadout.pb.h>

#include "Options.hpp"
#include "utils/Partitions.hpp"

#include <functional>


namespace fort {
namespace artemis {

class TrophallaxisDetector {
public :
	TrophallaxisDetector(size_t maxParallel, const cv::Size & size, const TrophallaxisOptions & options);


	void Detect(const cv::Mat & mat,
	            //size_t nThreads,
				//bool useCUDA,
	            hermes::FrameReadout & m);

	void testimwrite(cv::Mat image);

	int test = 0;

private:

	std::string	trophallaxismodel;
	std::string	labelfile;
	bool 		useCUDA;
	size_t		trophallaxisthreads;
};


} // namespace artemis
} // namespace fort
