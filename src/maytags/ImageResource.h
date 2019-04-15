#pragma once

#include <opencv2/core.hpp>

class ImageResource {
public:
	ImageResource(const uint8_t *start, const uint8_t * end);

	const cv::Mat & Image();
	void ExpectEqual(const cv::Mat & mat);
private :
	cv::Mat d_image;

	void ExpectEqual8uc1(const cv::Mat & mat);
	void ExpectEqual8uc3(const cv::Mat & mat);
};

#define LOAD_IMAGE(x) [](){	  \
		extern const uint8_t _binary_ ## x ## _start, _binary_ ## x ## _end; \
		return ImageResource(&_binary_ ## x ## _start, & _binary_ ## x ## _end); \
	}()
