#include "Apriltag2Process.h"

#include <apriltag/tag16h5.h>
#include <apriltag/tag25h7.h>
#include <apriltag/tag25h9.h>
#include <apriltag/tag36artoolkit.h>
#include <apriltag/tag36h10.h>
#include <apriltag/tag36h11.h>

#include <cmath>

#include <google/protobuf/util/delimited_message_util.h>

#include <asio/write.hpp>
#include <asio/connect.hpp>
#include <asio/posix/stream_descriptor.hpp>


#include <glog/logging.h>

#include <opencv2/imgcodecs.hpp>


#include "utils/PosixCall.h"


std::shared_ptr<asio::ip::tcp::socket> Connect(asio::io_service & io, const std::string & address) {
	asio::ip::tcp::resolver resolver(io);
	asio::ip::tcp::resolver::query q(address.c_str(),"artemis");
	asio::ip::tcp::resolver::iterator endpoint = resolver.resolve(q);
	auto res = std::make_shared<asio::ip::tcp::socket>(io);
	asio::connect(*res,endpoint);
}



AprilTag2Detector::AprilTag2Detector(const AprilTag2Detector::Config & config)
	: d_family(OpenFamily(config.Family))
	, d_detector(apriltag_detector_create(),apriltag_detector_destroy) {
	apriltag_detector_add_family(d_detector.get(),d_family.get());
	d_detector->nthreads = 1;

	d_detector->quad_decimate = config.QuadDecimate;
	d_detector->quad_sigma = config.QuadSigma;
	d_detector->refine_edges = config.RefineEdges ? 1 : 0;
	d_detector->refine_decode = config.NoRefineDecode ? 0 : 1;
	d_detector->refine_pose = config.RefinePose ? 1 : 0;
	d_detector->debug = false;
	d_detector->qtp.min_cluster_pixels = config.QuadMinClusterPixel;
	d_detector->qtp.max_nmaxima = config.QuadMaxNMaxima;
	d_detector->qtp.critical_rad = config.QuadCriticalRadian;
	d_detector->qtp.max_line_fit_mse = config.QuadMaxLineMSE;
	d_detector->qtp.min_white_black_diff = config.QuadMinBWDiff;
	d_detector->qtp.deglitch = config.QuadDeglitch ? 1 : 0;

}

AprilTag2Detector::~AprilTag2Detector() {}


AprilTag2Detector::FamilyPtr AprilTag2Detector::OpenFamily(const std::string & name ) {
	struct FamilyInterface {
		typedef apriltag_family_t* (*Constructor) ();
		typedef void (*Destructor) (apriltag_family_t *);
		Constructor c;
		Destructor  d;
	};

	static std::map<std::string,FamilyInterface > familyFactory = {
		{"16h5",{.c = tag16h5_create, .d=tag16h5_destroy}},
		{"25h7",{.c =tag25h7_create, .d=tag25h7_destroy}},
		{"25h9",{.c =tag25h9_create, .d=tag25h9_destroy}},
		{"36artoolkit",{.c =tag36artoolkit_create, .d=tag36artoolkit_destroy}},
		{"36h10",{.c =tag36h10_create, .d=tag36h10_destroy}},
		{"36h11",{.c =tag36h11_create, .d=tag36h11_destroy}}
	};

	auto fi = familyFactory.find(name);
	if (fi == familyFactory.end() ) {
		throw std::runtime_error("Could not find tag family '"+ name +"'");
	}
	return FamilyPtr(fi->second.c(),fi->second.d);
}



cv::Rect computeROI(size_t i, size_t nbROI, const cv::Size & imageSize, size_t margin) {
	size_t bandWidth = imageSize.width/nbROI;
	size_t leftMargin = (i == 0 ) ? 0 : margin;
	size_t rightMargin = (i == (nbROI - 1) ) ? 0 : margin;
	cv::Point2d roiOrig ((i*bandWidth) - leftMargin ,0);
	cv::Size roiSize(bandWidth + leftMargin + rightMargin, imageSize.height);
	return cv::Rect(roiOrig,roiSize);
}

AprilTag2Detector::ROITagDetection::ROITagDetection(const AprilTag2Detector::Ptr & parent)
	: d_parent(parent ) {
}

AprilTag2Detector::ROITagDetection::~ROITagDetection() {}

std::vector<ProcessFunction> AprilTag2Detector::ROITagDetection::Prepare(size_t maxProcess, const cv::Size & size ) {
	d_parent->d_results.resize(maxProcess);
	d_parent->d_offsets.resize(maxProcess);

	std::vector<ProcessFunction> toReturn;
	toReturn.reserve(maxProcess);
	for (size_t i = 0; i < maxProcess; ++i) {
		d_parent->d_results[i] = NULL;
		size_t margin = 75;
		cv::Rect roi = computeROI(i,maxProcess,size,margin);
		d_parent->d_offsets[i] = roi.x;

		toReturn.push_back([this,i,roi](const Frame::Ptr & frame,
		                                const cv::Mat & upstream,
		                                fort::FrameReadout & readout,		                                                                            cv::Mat & result) {
			                   cv::Mat withROI(frame->ToCV(),roi);
			                   image_u8_t img = {
				                   .width = withROI.cols,
				                   .height = withROI.rows,
				                   .stride = (int32_t) withROI.step[0],
				                   .buf = withROI.data
			                   };

			                   d_parent->d_results[i] = apriltag_detector_detect(d_parent->d_detector.get(),&img);
		                   });
	}
	return toReturn;

}

AprilTag2Detector::TagMerging::TagMerging(const AprilTag2Detector::Ptr & parent)
	: d_parent(parent){
}

AprilTag2Detector::TagMerging::~TagMerging() {}

std::vector<ProcessFunction> AprilTag2Detector::TagMerging::Prepare(size_t maxProcess, const cv::Size & size) {
	return { [this](const Frame::Ptr & frame,
	                const cv::Mat & upstream,
	                fort::FrameReadout & readout,
	                cv::Mat & result) {
			std::map<int32_t,apriltag_detection_t*> results;

			for(size_t i = 0; i < d_parent->d_results.size(); ++i ) {
				for( int j = 0; j < zarray_size(d_parent->d_results[i]); ++j ) {
					apriltag_detection_t * q;
					zarray_get(d_parent->d_results[i],j,&q);
					if (results.count((int32_t)q->id) != 0 ) {
						continue;
					}
					results[q->id] = q;
					q->c[0] += d_parent->d_offsets[i];
				}
			}
			readout.set_timestamp(frame->Timestamp());
			readout.set_frameid(frame->ID());
			readout.clear_ants();
			auto time = readout.mutable_time();
			time->set_seconds(frame->Time().tv_sec);
			time->set_nanos(frame->Time().tv_usec * 1000);
			for ( auto & kv : results ) {
				auto a = readout.add_ants();
				a->set_id(kv.second->id);
				a->set_x(kv.second->c[0]);
				a->set_y(kv.second->c[1]);
				//TODO: compute angle!!!
				a->set_theta(0.0);
			}

			for(size_t i = 0 ; i < d_parent->d_results.size(); ++ i ) {
				apriltag_detections_destroy(d_parent->d_results[i]);
			}

			//clear any upstream transformation
			result = frame->ToCV();
		}
	};
}


AprilTag2Detector::Finalization::Finalization(const AprilTag2Detector::Ptr & parent,
                                              asio::io_service & service,
                                              const std::string & address,
                                              const std::string & savePath,
                                              size_t newAntROISize)
	: d_parent(parent)
	, d_service(service)
	, d_address(address)
	, d_strand(asio::strand(service))
	, d_savePath(savePath)
	, d_newAntROISize(newAntROISize) {

	auto prodConsum = BufferPool::Create();
	d_producer = std::move(prodConsum.first);
	d_consumer = std::move(prodConsum.second);

	if (!d_address.empty()) {
		try {
			d_socket = Connect(d_service,d_address);
		} catch ( const std::exception & e) {
			LOG(ERROR) << "Could not connect to '" << d_address << "': " << e.what();
			ScheduleReconnect();
		}
	}
}

AprilTag2Detector::Finalization::~Finalization() {};

std::vector<ProcessFunction> AprilTag2Detector::Finalization::Prepare(size_t maxProcess, const cv::Size &) {
	if (maxProcess == 1) {
		return {[this](const Frame::Ptr & frame,
		               const cv::Mat & upstream,
		               fort::FrameReadout & readout,
		               cv::Mat & result){
				SerializeMessage(readout);
				CheckForNewAnts(frame,readout,0,1);
			}};
	}

	std::vector<ProcessFunction> res;
	res.reserve(maxProcess);
	res.push_back([this](const Frame::Ptr & frame,
	                     const cv::Mat & upstream,
	                     fort::FrameReadout & readout,
	                     cv::Mat & result) {
		              SerializeMessage(readout);
	              });
	// uncomment to avoid multi-threading
	// maxProcess = 2;
	for (size_t i = 1; i < maxProcess; ++i) {
		res.push_back([this,i,maxProcess](const Frame::Ptr & frame,
		                                  const cv::Mat & upstream,
		                                  fort::FrameReadout & readout,
		                                  cv::Mat & result) {
			              CheckForNewAnts(frame,readout,i-1,maxProcess-1);
		              });
	};
	return res;
}

void AprilTag2Detector::Finalization::SerializeMessage(const fort::FrameReadout & message) {
	if (!d_socket) {
		LOG(WARNING) << "serialization: discarding data: no socket available";
		return;
	}

	if ( d_producer->Full() ) {
		LOG(WARNING) << "serialization: discarding data: FIFO full";
	}

	asio::streambuf & buf = d_producer->Tail();
	buf.consume(buf.max_size());
	std::ostream output(&buf);
	google::protobuf::util::SerializeDelimitedToOstream(message, &output);
	d_producer->Push();
	d_strand.wrap([this]{
			if(d_consumer->Empty()) {
				return;
			}
			asio::async_write(*d_socket,
			                  d_consumer->Head().data(),
			                  [this](const asio::error_code & ec,
			                         std::size_t){
				                  d_consumer->Pop();
				                  if (ec == asio::error::connection_reset || ec == asio::error::bad_descriptor ) {
					                  if (!d_socket) {
						                  return;
					                  }
					                  d_socket.reset();
					                  ScheduleReconnect();
				                  }
			                  });
		});

}

void AprilTag2Detector::Finalization::ScheduleReconnect() {
	if (d_socket) {
		return;
	}
	LOG(INFO) << "Reconnecting in 5s to '" << d_address << "'";
	auto t = std::make_shared<asio::deadline_timer>(d_service,boost::posix_time::seconds(5));
	t->async_wait([this,t](const asio::error_code & ) {
			LOG(INFO) << "Reconnecting to '" << d_address << "'";
			try {
				d_socket = Connect(d_service,d_address);
			} catch ( const std::exception & e) {
				LOG(ERROR) << "Could not connect to '" << d_address << "':  " << e.what();
				ScheduleReconnect();
			}
		});
}


void AprilTag2Detector::Finalization::CheckForNewAnts( const Frame::Ptr & frame,
                                                       const fort::FrameReadout & readout,
                                                       size_t start,
                                                       size_t stride) {
	for (size_t i = start; i < readout.ants_size(); i += stride) {
		auto a = readout.ants(i);
		int32_t ID = a.id();
		{
			std::lock_guard<std::mutex> lock(d_mutex);
			if ( d_known.count(a.id()) != 0 ) {
				continue;
			}
		}
		cv::Rect roi(cv::Point2d(((size_t)a.x())-d_newAntROISize/2,
		                         ((size_t)a.y())-d_newAntROISize/2),
		             cv::Size(d_newAntROISize,
		                      d_newAntROISize));

		auto pngData =  std::make_shared<std::vector<uint8_t> >();
		cv::imencode("png",cv::Mat(frame->ToCV(),roi),*pngData);

		d_service.post([this,pngData,ID](){
				std::ostringstream oss(d_savePath);
				oss << "/ant_" << ID << ".png";
				int fd = open(oss.str().c_str(), O_CREAT|O_WRONLY| O_NONBLOCK);
				if (fd == -1) {
					LOG(ERROR) << "Could not save ant " << ID << ": " << std::error_code(errno,ARTEMIS_SYSTEM_CATEGORY());
					return;
				}
				auto stream = std::make_shared<asio::posix::stream_descriptor>(d_service,fd);
				asio::async_write(*stream,
				                  asio::const_buffers_1(&((*pngData)[0]),pngData->size()),
				                  [this,pngData,stream](const asio::error_code & ec,
				                                        std::size_t ) {
					                  close(stream->native_handle());
				                  });
			});
	}
}

ProcessQueue AprilTag2Detector::Create(const Config & config,
                                       asio::io_service & service,
                                       const std::string & address,
                                       const std::string & path) {
	auto detector = std::shared_ptr<AprilTag2Detector>(new AprilTag2Detector(config));
	return {
		std::shared_ptr<ProcessDefinition>(new ROITagDetection(detector)),
		std::shared_ptr<ProcessDefinition>(new TagMerging(detector)),
		std::shared_ptr<ProcessDefinition>(new Finalization(detector,service,address,path,config.NewAntROISize))
	};
}
