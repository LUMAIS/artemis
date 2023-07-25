// it look like ApriltagDetector.cpp

#include "TrackingDetector.hpp"
#include "detectors/Tracking.hpp"
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

namespace fort
{
	namespace artemis
	{

		std::vector<ALObject> objects;

		TrackingDetector::TrackingDetector(size_t maxParallel, const cv::Size &size, const TrackingOptions &options)
		{
			DLOG(INFO) << "[TrackingDetector]: Started";

			trackingmodel = options.trackingmodel;
			labelfile = options.labelfile;
			useCUDA = options.useCUDA;
			trackingthreads = options.trackingthreads;
		}

		void TrackingDetector::Detect(const cv::Mat &image, /*size_t nThreads, bool useCUDA,*/ hermes::FrameReadout &m)
		{

			bool usemodel = false;

			if (trackingmodel != "none")
			{
				try
				{
					// Deserialize the ScriptModule from a file using torch::jit::load().
					torch::jit::script::Module module;
					module = torch::jit::load(trackingmodel);
					usemodel = true;
				}
				catch (const c10::Error &e)
				{
					std::cerr << "error loading the model\n";
				}
			}

			std::vector<std::pair<cv::Point2f, uint16_t>> detects;

			torch::DeviceType device_type;

			if (useCUDA == true)
				device_type = torch::kCUDA;
			else
				device_type = torch::kCPU;

			// DLOG(INFO) << "[TrackingDetector]: DeviceType - " << device_type;

			detects = framePrev.empty()
				? DetectorMotionV2_1(trackingmodel, device_type, image, image, objects, id_frame, usemodel)
				: DetectorMotionV2_1(trackingmodel, device_type, framePrev, image, objects, id_frame, usemodel);
			framePrev = image;
			++id_frame;

			for (uint16_t i = 0; i < detects.size(); i++)
			{
				auto t = m.add_tags();
				t->set_x(detects[i].first.x);
				t->set_y(detects[i].first.y);
				t->set_id(detects[i].second);
				// t->set_theta(angle);
			}
		}

	} // namespace artemis
} // namespace fort
