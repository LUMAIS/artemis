//it look like ApriltagDetector.cpp

#include "TrophallaxisDetector.hpp"
#include "trophallaxis/antdetect.hpp"

#include <fort/tags/fort-tags.h>

#include <tbb/parallel_for.h>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <torch/script.h> 


#include <glog/logging.h>
#include <vector>
#include <unistd.h>

namespace fort {
namespace artemis {

TrophallaxisDetector::TrophallaxisDetector(size_t maxParallel, const cv::Size & size, const TrophallaxisOptions & options) {
	DLOG(INFO) << "[TrophallaxisDetector]: Started";

	trophallaxismodel = options.trophallaxismodel;
	labelfile = options.labelfile;
	useCUDA = options.useCUDA;
	trophallaxisthreads = options.trophallaxisthreads;
}

void TrophallaxisDetector::Detect(const cv::Mat & image, /*size_t nThreads, bool useCUDA,*/ hermes::FrameReadout & m) {

	//std::cout<<"TrophallaxisDetector | Detect | nThreads - "<<nThreads<<std::endl;
	//nThreads = std::min(std::max(nThreads,size_t(1)),d_detectors.size());
	/*
	std::vector<std::array<float,2>> testvector;
	testvector = TDetect(useCUDA,trophallaxismodel,labelfile, image, trophallaxisthreads);
	testimwrite(image);
	*/

	torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(trophallaxismodel);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }

	std::vector<cv::Point2f> detects;

	torch::DeviceType device_type = torch::kCPU;
	detects = detectorT (module, image,device_type);

	for(int i =0; i<detects.size(); i++)
	{
		auto t = m.add_tags();
		t->set_x(detects[i].x);
		t->set_y(detects[i].y);
	}
	
	/*
	auto t = m.add_tags();
		//t->set_id(tagID);
		t->set_x(1000 + test*5);
		t->set_y(800 + test*10);
		//t->set_theta(angle);
	*/

}
} // namespace artemis
} // namespace fort
