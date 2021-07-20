//it look like ApriltagDetector.cpp
//#include "testtorch.h"

#include "TrophallaxisDetector.hpp"
#include "trophallaxis/antdetect.hpp"

#include <fort/tags/fort-tags.h>

#include <tbb/parallel_for.h>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <glog/logging.h>



namespace fort {
namespace artemis {


TrophallaxisDetector::TrophallaxisDetector(size_t maxParallel, const cv::Size & size, const TrophallaxisOptions & options) {
	DLOG(INFO) << "[TrophallaxisDetector]: Started";

	trophallaxismodel = options.trophallaxismodel;
	labelfile = options.labelfile;
	useCUDA = options.useCUDA;
	trophallaxisthreads = options.trophallaxisthreads;



	//std::cout<<"options.trophallaxismodel -> "<<options.trophallaxismodel<<std::endl;
}


void TrophallaxisDetector::testimwrite(cv::Mat image){

	//cv::Size imageSize(500,500);
    //cv::Mat imageD(imageSize, CV_8UC3, cv::Scalar(155, 11, 250));

	cv::Mat bufimage;
	cv::resize(image, bufimage,cv::Size(300, 300),cv::InterpolationFlags::INTER_CUBIC);

	//imshow("Detector", bufimage);
    //cv::waitKey(5);
	
}

void TrophallaxisDetector::Detect(const cv::Mat & image, /*size_t nThreads, bool useCUDA,*/ hermes::FrameReadout & m) {

	//std::cout<<"TrophallaxisDetector | Detect | nThreads - "<<nThreads<<std::endl;
	//nThreads = std::min(std::max(nThreads,size_t(1)),d_detectors.size());

	std::vector<std::array<float,2>> testvector;

	testvector = TDetect(useCUDA,trophallaxismodel,labelfile, image, trophallaxisthreads);

	std::cout<<"testvector.x - "<<testvector.at(0)[0]<<std::endl;//" | testvector.y - "<<testvector<<std::endl;

	testimwrite(image);
	
	auto t = m.add_tags();
		//t->set_id(tagID);
		t->set_x(1000 + test*5);
		t->set_y(800 + test*10);
		//t->set_theta(angle);

	auto s = m.tags(0);
	std::cout<<"x - "<<s.x()<<std::endl;

    test++;
	
}

//////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////

} // namespace artemis
} // namespace fort
