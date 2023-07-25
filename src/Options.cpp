#include "Options.hpp"
#include "fort/time/Time.hpp"

#include <stdexcept>


#include <utils/FlagParser.hpp>
#include <utils/StringManipulation.hpp>

#include <dirent.h>
#include <algorithm>
#include <fnmatch.h>

namespace fort {
namespace artemis {

std::vector<uint32_t> ParseCommaSeparatedListHexa(std::string & list) {
	if ( list.empty() ) {
		return {};
	}
	std::vector<uint32_t> res;
	std::vector<std::string> tagIDs;
	base::SplitString(list.cbegin(),
	                  list.cend(),
	                  ",",
	                  std::back_inserter<std::vector<std::string>>(tagIDs));
	res.reserve(tagIDs.size());
	for (auto tagIDStr : tagIDs) {
		if ( base::HasPrefix(base::TrimSpaces(tagIDStr),"0x") == false ) {
			throw std::runtime_error("'" + tagIDStr + "' is not an hexadecimal number");
		}
		std::istringstream is(base::TrimSpaces(tagIDStr));
		uint32_t tagID;
		is >> std::hex >> tagID;
		if ( !is.good() && is.eof() == false ) {
			std::ostringstream os;
			os << "Cannot parse '" << tagIDStr << "'  in  '" << list << "'";
			throw std::runtime_error(os.str());
		}
		res.push_back(tagID);
	}
	return res;
}

std::vector<size_t> ParseCommaSeparatedList(std::string & list) {
	if ( list.empty() ) {
		return {};
	}
	std::vector<size_t> res;
	std::vector<std::string> IDs;
	base::SplitString(list.cbegin(),
	                  list.cend(),
	                  ",",
	                  std::back_inserter<std::vector<std::string>>(IDs));
	res.reserve(IDs.size());
	for (auto IDstr : IDs) {
		std::istringstream is(base::TrimSpaces(IDstr));
		uint64_t ID;
		is >> ID;
		if ( !is.good() && is.eof() == false ) {
			std::ostringstream os;
			os << "Cannot parse '" << IDstr << "'  in  '" << list << "'";
			throw std::runtime_error(os.str());
		}
		res.push_back(ID);
	}
	return res;
}

fort::tags::Family ParseTagFamily(const std::string & f) {
	static std::map<std::string,fort::tags::Family> families
		= {
		   {"",fort::tags::Family::Undefined},
		   {"16h5",fort::tags::Family::Tag16h5},
		   {"25h9",fort::tags::Family::Tag25h9},
		   {"36h10",fort::tags::Family::Tag36h10},
		   {"36h11",fort::tags::Family::Tag36h11},
		   {"36ARTag",fort::tags::Family::Tag36ARTag},
		   {"Circle21h7",fort::tags::Family::Circle21h7},
		   {"Circle49h12",fort::tags::Family::Circle49h12},
		   {"Custom48h12",fort::tags::Family::Custom48h12},
		   {"Standard41h12",fort::tags::Family::Standard41h12},
		   {"Standard52h13",fort::tags::Family::Standard52h13},
	};
	auto fi = families.find(f);
	if ( fi == families.end() ) {
		throw std::out_of_range("Umknown family '" + f + "'");
	}
	return fi->second;
}

GeneralOptions::GeneralOptions()
	: PrintHelp(false)
	, PrintVersion(false)
	, PrintResolution(false)
	, LogDir("")
	, TestMode(false)
	, LegacyMode(false) {
}

void GeneralOptions::PopulateParser(options::FlagParser & parser) {
	parser.AddFlag("help",PrintHelp,"Print this help message",'h');
	parser.AddFlag("fetch-resolution",PrintResolution,"Print the camera resolution");
	parser.AddFlag("version",PrintVersion,"Print version");
	parser.AddFlag("log-output-dir",LogDir,"Directory to put logs in");
	parser.AddFlag("stub-image-paths", stubImagePaths, "Use a suite of stub images instead of an actual framegrabber");
	parser.AddFlag("input-frames", inputImagePathsMask, "Use a suite of input images instead of an actual framegrabber");
	parser.AddFlag("input-video", inputVideoPath, "Use of input video instead of an actual framegrabber");
	parser.AddFlag("test-mode",TestMode,"Test mode, adds an overlay detection drawing and statistics");
	parser.AddFlag("legacy-mode",LegacyMode,"Uses a legacy mode data output for ants cataloging and video output display. The data will be convertible to the data expected by the former Keller's group tracking system");
}

//Example PathsMask "sources/100testimages/t10_smallpopu_*.bmp"
std::vector<std::string> GetFramesPaths(std::string PathsMask){

	std::vector<std::string> fpaths;
	std::string::size_type maskPos = PathsMask.rfind('/');
    
	if (maskPos != std::string::npos)
        ++maskPos;
    else
        maskPos = 0;

    std::string path = PathsMask.substr(0,maskPos);
    std::string mask = PathsMask.substr(maskPos);

    struct dirent **namelist;
    int n;

    n = scandir(path.c_str(), &namelist, NULL, alphasort);

    if (n < 0)
        perror("scandir");
    else {
        while (n--) {

        if (fnmatch(mask.c_str(), namelist[n]->d_name, FNM_CASEFOLD) == 0)
			fpaths.push_back(path+namelist[n]->d_name);

        	free(namelist[n]);
        }
        free(namelist);
    }
    std::sort(fpaths.begin(),fpaths.end());
	return fpaths;
}

void GeneralOptions::FinishParse() {
	base::SplitString(stubImagePaths.cbegin(),
	                  stubImagePaths.cend(),
	                  ",",
	                  std::back_inserter<std::vector<std::string>>(StubImagePaths));

	if(inputImagePathsMask.length() > 0)
		StubImagePaths = GetFramesPaths(inputImagePathsMask);
	
}

NetworkOptions::NetworkOptions()
	: Host()
	, Port(3002) {
}

void NetworkOptions::PopulateParser(options::FlagParser & parser) {
	parser.AddFlag("host", Host, "Host to send tag detection readout");
	parser.AddFlag("port", Port, "Port to send tag detection readout",'p');
}

void NetworkOptions::FinishParse() {}

VideoOutputOptions::VideoOutputOptions()
	: Height(1080)
	, AddHeader(false)
	, ToStdout(false)
{
}

void VideoOutputOptions::PopulateParser(options::FlagParser & parser) {
	// General options
	parser.AddFlag("video-output-height", Height, "Video Output height, width computed to maintain the aspect ratio, 0 means use frame height ");
	parser.AddFlag("video-output-file", ToFile, "Sends video output to file(s) with this basename, automatically adding the suffix \"_CamId-<CamId>.mp4\" ");
	// Video stdout options
	parser.AddFlag("video-output-stdout", ToStdout, "Sends video output to stdout");
	parser.AddFlag("video-output-stdout-header", AddHeader, "Adds binary header to stdout output");
}

void VideoOutputOptions::FinishParse() {
}

cv::Size VideoOutputOptions::WorkingResolution(const cv::Size & input) const {
	return Height ? cv::Size(input.width * double(Height) / double(input.height),Height) : input;
}

DisplayOptions::DisplayOptions()
	: NoGUI(false)
	// , RenderHeight(0)
{
}

void DisplayOptions::PopulateParser(options::FlagParser & parser) {
	parser.AddFlag("no-gui", NoGUI, "Disable GUI");
	// parser.AddFlag("rendering-height", RenderHeight, "Rendering height of OpenCV windows");
	parser.AddFlag("highlight-tags",d_highlighted,"Tag to highlight when drawing detections");
}

void DisplayOptions::FinishParse() {
	Highlighted = ParseCommaSeparatedListHexa(d_highlighted);
}


std::string formatDuration(const fort::Duration & d ) {
	std::ostringstream oss;
	oss << d;
	return oss.str();
}

ProcessOptions::ProcessOptions()
	: FrameStride(1)
	, FrameID()
	, UUID()
	, NewAntOutputDir()
	, AntTraceFile()
	, NewAntROISize(600)
	, ImageRenewPeriod(2 * Duration::Hour) {
	d_imageRenewPeriod = formatDuration(ImageRenewPeriod);
}

void ProcessOptions::PopulateParser(options::FlagParser & parser) {
	parser.AddFlag("frame-stride",FrameStride,"Frame sequence length");
	parser.AddFlag("frame-ids",d_frameIDs,"Frame ID to consider in the frame sequence, if empty consider all");
	parser.AddFlag("new-ant-output-dir",NewAntOutputDir,"Path where to save new detected ant pictures");
	parser.AddFlag("ant-tracing-file",AntTraceFile,"Outuput file in the CSV format for the ant tracking (i.e., runs/tacking.ssv)",'t');
	parser.AddFlag("new-ant-roi-size", NewAntROISize, "Size of the image to save when a new ant is found");
	parser.AddFlag("image-renew-period", d_imageRenewPeriod, "ant cataloguing and full frame export renew period");
	parser.AddFlag("uuid", UUID,"The UUID to mark data sent over network");
}

void ProcessOptions::FinishParse() {
	auto IDs = ParseCommaSeparatedList(d_frameIDs);
	FrameID.clear();
	FrameID.insert(IDs.begin(),IDs.end());
	ImageRenewPeriod = Duration::Parse(d_imageRenewPeriod);
}

CameraOptions::CameraOptions()
	: FPS(8.0)
	, Triggermode("none")
	, StrobeDuration(1500 * Duration::Microsecond)
	, StrobeDelay(0) 
	, SlaveWidth(0)  // Introduced by Serhii for slave cameras in DF mode
	, SlaveHeight(0)  // Introduced by Serhii
{
	d_strobeDuration = formatDuration(StrobeDuration);
	d_strobeDelay = formatDuration(StrobeDelay);
}

void CameraOptions::PopulateParser(options::FlagParser & parser)  {
	parser.AddFlag("camera-fps",FPS,"Camera FPS to use");
	parser.AddFlag("camera-id", cameraID, "Сamera ID");
	parser.AddFlag("trigger-mode", Triggermode, "Use a trigger to get a frame sequential/parallel");
	parser.AddFlag("camera-slave-width",SlaveWidth,"Camera Width argument for slave mode");
	parser.AddFlag("camera-slave-height",SlaveHeight,"Camera Height argument for slave mode");
	parser.AddFlag("camera-strobe",d_strobeDuration,"Camera Strobe duration");
	parser.AddFlag("camera-strobe-delay",d_strobeDelay,"Camera Strobe delay");
}

void CameraOptions::FinishParse() {
	StrobeDuration = Duration::Parse(d_strobeDuration);
	StrobeDelay = Duration::Parse(d_strobeDelay);
}

ApriltagOptions::ApriltagOptions()
	: Family(fort::tags::Family::Undefined)
	, QuadDecimate(1.0)
	, QuadSigma(0.0)
	, RefineEdges(false)
	, QuadMinClusterPixel(5)
	, QuadMaxNMaxima(10)
	, QuadCriticalRadian(0.174533)
	, QuadMaxLineMSE(10.0)
	, QuadMinBWDiff(40)
	, QuadDeglitch(false) {
 }

void ApriltagOptions::PopulateParser(options::FlagParser & parser)  {
	parser.AddFlag("at-quad-decimate",QuadDecimate,"Decimate original image for faster computation but worse pose estimation. Should be 1.0 (no decimation), 1.5, 2, 3 or 4");
	parser.AddFlag("at-quad-sigma",QuadSigma,"Apply a gaussian filter for quad detection, noisy image likes a slight filter like 0.8, for ant detection, 0.0 is almost always fine");
	parser.AddFlag("at-refine-edges",RefineEdges,"Refines the edge of the quad, especially needed if decimation is used.");
	parser.AddFlag("at-quad-min-cluster",QuadMinClusterPixel,"Minimum number of pixel to consider it a quad");
	parser.AddFlag("at-quad-max-n-maxima",QuadMaxNMaxima,"Number of candidate to consider to fit quad corner");
	parser.AddFlag("at-quad-critical-radian",QuadCriticalRadian,"Rejects quad with angle to close to 0 or 180 degrees");
	parser.AddFlag("at-quad-max-line-mse",QuadMaxLineMSE,"MSE threshold to reject a fitted quad");
	parser.AddFlag("at-quad-min-bw-diff",QuadMinBWDiff,"Difference in pixel value to consider a region black or white");
	parser.AddFlag("at-quad-deglitch",QuadDeglitch,"Deglitch only for noisy images");
	parser.AddFlag("at-family",d_family,"The apriltag family to use");
}

void ApriltagOptions::FinishParse() {
	Family = ParseTagFamily(d_family);
}

TrophallaxisOptions::TrophallaxisOptions() :
	 trophallaxismodel("")
	, labelfile("")
	, useCUDA(false)
	, trophallaxisthreads(1) {
 }

 void TrophallaxisOptions::PopulateParser(options::FlagParser & parser)  {
	parser.AddFlag("at-trophallaxis-model",trophallaxismodel,"The path to the trophallaxis model");
	parser.AddFlag("at-label-file",labelfile,"The path to the label file");
	parser.AddFlag("at-useCUDA",useCUDA,"Use CUDA?");
	parser.AddFlag("at-trophallaxis-threads",trophallaxisthreads,"Number of threads for trophallaxis detection");
}

TrackingOptions::TrackingOptions() :
	 trackingmodel("")
	, labelfile("")
	, useCUDA(false)
	, trackingthreads(1) {
 }

 void TrackingOptions::PopulateParser(options::FlagParser & parser)  {
	parser.AddFlag("at-tracking-model",trackingmodel,"The path to the tracking model");
	parser.AddFlag("at-label-file",labelfile,"The path to the label file");
	parser.AddFlag("at-useCUDA",useCUDA,"Use CUDA?");
	parser.AddFlag("at-tracking-threads",trackingthreads,"Number of threads for tracking detection");
}

void Options::PopulateParser(options::FlagParser & parser)  {
	General.PopulateParser(parser);
	Display.PopulateParser(parser);
	Network.PopulateParser(parser);
	VideoOutput.PopulateParser(parser);
	Apriltag.PopulateParser(parser);
	Trophallaxis.PopulateParser(parser);
	Tracking.PopulateParser(parser);
	Camera.PopulateParser(parser);
	Process.PopulateParser(parser);
}

void Options::FinishParse()  {
	General.FinishParse();
	Display.FinishParse();
	Network.FinishParse();
	VideoOutput.FinishParse();
	Apriltag.FinishParse();
	Camera.FinishParse();
	Process.FinishParse();
}

Options Options::Parse(int & argc, char ** argv, bool printHelp) {
	Options opts;
	options::FlagParser parser(options::FlagParser::Default,"low-level vision detection for the FORmicidae Tracker");

	opts.PopulateParser(parser);
	parser.Parse(argc,argv);
	opts.FinishParse();

	if ( opts.General.PrintHelp && printHelp ) {
		parser.PrintUsage(std::cerr);
		exit(0);
	}

	opts.Validate();


	return opts;
}


void Options::Validate() {

	if ( Process.FrameStride == 0 ) {
		Process.FrameStride = 1;
	}

	if ( Process.FrameID.empty() ) {
		for ( size_t i = 0; i < Process.FrameStride; ++i ) {
			Process.FrameID.insert(i);
		}
	}


	for ( const auto & frameID : Process.FrameID ) {
		if ( frameID >= Process.FrameStride ) {
			throw std::invalid_argument(std::to_string(frameID)
			                            + " is outside of frame stride range [0;"
			                            + std::to_string(Process.FrameStride) + "[");

		}
	}
#ifdef NDEBUG
	if ( Process.ImageRenewPeriod < 15 * Duration::Minute ) {
		throw std::invalid_argument("Image renew period (" + formatDuration(Process.ImageRenewPeriod) + ") is too small for production of large dataset (minimum: 15m)");
	}
#endif



}

} // namespace artemis
} // namespace fort
