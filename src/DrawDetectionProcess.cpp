#include "DrawDetectionProcess.h"

#include <opencv2/imgproc.hpp>

#include <Eigen/Geometry>

std::vector<ProcessFunction> DrawDetectionProcess::Prepare(size_t maxProcess, const cv::Size &) {
	std::vector<ProcessFunction> res;
	for( size_t i = 0 ; i< maxProcess; ++ i) {
		res.push_back([i,maxProcess](const Frame::Ptr &,
		                 const cv::Mat &,
		                 fort::FrameReadout & readout,
		                 cv::Mat & result) {
			              DrawAnts(i,maxProcess,readout,result);
		              });
	}
}

void DrawDetectionProcess::DrawAnts(size_t start, size_t stride, const fort::FrameReadout & readout,cv::Mat & result) {
	for (size_t i = start; i < readout.ants_size(); i += stride ) {
		DrawAnt(readout.ants(i),result,50);
	}
}
void DrawDetectionProcess::DrawAnt(const fort::Ant & a, cv::Mat & result, size_t size) {

	Eigen::Vector2d top(0,-size*2.0/3.0),left(-size/2.0,size/3.0),right(size/2.0,size/3.0),center(a.x(),a.y());

	Eigen::Rotation2D<double> rot(a.theta());
	top = rot * top + center;
	left = rot * left + center;
	right = rot * right + center;

	cv::line(result,cv::Point(top(0),top(1)),cv::Point(left(0),left(1)),cv::Scalar(0,0,0xff),2);
	cv::line(result,cv::Point(top(0),top(1)),cv::Point(right(0),right(1)),cv::Scalar(0,0,0xff),2);
	cv::line(result,cv::Point(left(0),left(1)),cv::Point(right(0),right(1)),cv::Scalar(0,0,0xff),2);
	std::ostringstream oss;
	oss << a.id();
	int fontface = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
	double fontscale = 1.0;
	int baseline;
	cv::Size textsize = cv::getTextSize(oss.str(), fontface, fontscale, 2,
	                                    &baseline);

	cv::putText(result,
	            oss.str(),
	            cv::Point(center(0)-textsize.width/2,
	                      center(1)+textsize.height/2 + size),
	            fontface,
	            fontscale,
	            cv::Scalar(0x00, 0x00, 0xff),
	            2);
}
