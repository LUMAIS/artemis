#include "ResizeProcess.h"

#include <opencv2/imgproc.hpp>

ResizeProcess::ResizeProcess(size_t height)
	: d_height(height)
	, d_initialized(false) {
}

ResizeProcess::~ResizeProcess() {}


std::vector<ProcessFunction> ResizeProcess::Prepare(size_t nbProcess, const cv::Size & size) {
	if (d_initialized == false) {
		size_t width = (size.width * d_height) / size.height ;
		d_resized = cv::Mat(d_height,width,CV_8U);
		d_initialized = true;
	}

	return {
		[this](const Frame::Ptr & frame,
		             const cv::Mat & upstream,
		             fort::FrameReadout & readout,
		             cv::Mat & result) {
			cv::resize(upstream,d_resized,d_resized.size());
			cv::cvtColor(d_resized,result,CV_GRAY2RGB);
		}
	};
}