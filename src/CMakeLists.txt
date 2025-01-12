include(CheckCXXSourceRuns)

Check_CXX_Source_Runs(
"
#include <system_error>


int main() {
	if (std::error_code(EDOM,std::system_category()) == std::errc::argument_out_of_domain ) {
		return 0;
	}
	return 1;
}
"
ARTEMIS_USE_SYSTEM_CATEGORY)

configure_file(config.h.in artemis-config.h)
################ONNX RUNTIME################
#set(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime/onnxruntime-linux-x64-1.8.1/include/)
#find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h onnxruntime/onnxruntime-linux-x64-1.8.1/include/)
#find_path(ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS cuda_provider_factory.h HINTS onnxruntime/onnxruntime-linux-x64-gpu-1.8.1/include/)
#find_library(ONNX_RUNTIME_LIB onnxruntime HINTS onnxruntime/onnxruntime-linux-x64-gpu-1.8.1/lib/)
############################################

include_directories(${CMAKE_CURRENT_BINARY_DIR}
	                ${CMAKE_CURRENT_SOURCE_DIR}
					################ONNX RUNTIME################
					#${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} 
					#${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS} 
					#${ONNX_VERSION_CONVERTER_INCLUDE_DIRS}
					############################################
	                )

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ui/shaders_data.c ${CMAKE_CURRENT_BINARY_DIR}/ui/shaders_data.h
                   COMMAND ${RESOURCIFY_EXECUTABLE} ARGS "-O" ${CMAKE_CURRENT_BINARY_DIR}/ui/shaders_data
                                                         "-I" ${CMAKE_CURRENT_SOURCE_DIR}/ui/shaders/frame.vertexshader
                                                         "-I" ${CMAKE_CURRENT_SOURCE_DIR}/ui/shaders/frame.fragmentshader
                                                         "-I" ${CMAKE_CURRENT_SOURCE_DIR}/ui/shaders/primitive.vertexshader
                                                         "-I" ${CMAKE_CURRENT_SOURCE_DIR}/ui/shaders/primitive.fragmentshader
                                                         "-I" ${CMAKE_CURRENT_SOURCE_DIR}/ui/shaders/circle.fragmentshader
                                                         "-I" ${CMAKE_CURRENT_SOURCE_DIR}/ui/shaders/font.vertexshader
                                                         "-I" ${CMAKE_CURRENT_SOURCE_DIR}/ui/shaders/font.fragmentshader
                                                         "-I" ${CMAKE_CURRENT_SOURCE_DIR}/ui/shaders/roi.fragmentshader


                   DEPENDS resourcify
                           ${CMAKE_CURRENT_SOURCE_DIR}/ui/shaders/frame.vertexshader
                           ${CMAKE_CURRENT_SOURCE_DIR}/ui/shaders/frame.fragmentshader
                           ${CMAKE_CURRENT_SOURCE_DIR}/ui/shaders/primitive.vertexshader
                           ${CMAKE_CURRENT_SOURCE_DIR}/ui/shaders/circle.fragmentshader
                           ${CMAKE_CURRENT_SOURCE_DIR}/ui/shaders/font.vertexshader
                           ${CMAKE_CURRENT_SOURCE_DIR}/ui/shaders/font.fragmentshader
                           ${CMAKE_CURRENT_SOURCE_DIR}/ui/shaders/primitive.fragmentshader
                           ${CMAKE_CURRENT_SOURCE_DIR}/ui/shaders/roi.fragmentshader
                           )

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/fonts/fonts_data.c ${CMAKE_CURRENT_BINARY_DIR}/fonts/fonts_data.h
                   COMMAND ${RESOURCIFY_EXECUTABLE} ARGS "-O" ${CMAKE_CURRENT_BINARY_DIR}/fonts/fonts_data
                                                         "-I" ${CMAKE_CURRENT_SOURCE_DIR}/fonts/vga.fon

                   DEPENDS resourcify
                           ${CMAKE_CURRENT_SOURCE_DIR}/fonts/vga.fon
                           )

set(SRC_FILES utils/Defer.cpp
	          utils/PosixCall.cpp
	          utils/FlagParser.cpp
	          utils/StringManipulation.cpp
	          utils/Partitions.cpp
	          Task.cpp
	          Options.cpp
	          FrameGrabber.cpp
	          StubFrameGrabber.cpp
			  StubVideoGrabber.cpp
	          Connection.cpp
	          Application.cpp
	          AcquisitionTask.cpp
	          ProcessFrameTask.cpp
	          ApriltagDetector.cpp
			  TrophallaxisDetector.cpp
			  TrackingDetector.cpp
	          FullFrameExportTask.cpp
	          UserInterfaceTask.cpp
	          VideoStdoutTask.cpp
	          ImageTextRenderer.cpp
	          ui/UserInterface.cpp
	          ui/StubUserInterface.cpp
	          ui/GLVertexBufferObject.cpp
	          ui/GLUserInterface.cpp
	          ui/ShaderUtils.cpp
	          ui/shaders_data.c
	          fonts/fonts_data.c
	          ui/GLFont.cpp
	          Utils.cpp
	          main.cpp
	          )


set(HDR_FILES utils/Defer.hpp
              utils/PosixCall.hpp
              utils/PosixCall.hpp
	          utils/FlagParser.hpp
	          utils/StringManipulation.hpp
	          utils/Partitions.hpp
	          Task.hpp
	          FrameGrabber.hpp
	          Connection.hpp
	          StubFrameGrabber.hpp
			  StubVideoGrabber.hpp
	          Options.hpp
	          AcquisitionTask.hpp
	          ProcessFrameTask.hpp
	          ApriltagDetector.hpp
			  TrophallaxisDetector.hpp
			  TrackingDetector.hpp
	          FullFrameExportTask.hpp
	          UserInterfaceTask.hpp
	          VideoStdoutTask.hpp
	          ImageTextRenderer.hpp
	          ui/UserInterface.hpp
	          ui/StubUserInterface.hpp
	          ui/GLVertexBufferObject.hpp
	          ui/GLUserInterface.hpp
	          ui/ShaderUtils.hpp
	          ui/shaders_data.h
	          fonts/fonts_data.h
	          ui/GLFont.hpp
	          Application.hpp
	          Utils.hpp
	          )


set(UTEST_SRC_FILES main-check.cpp
                    utils/DeferUTest.cpp
                    utils/FlagParserUTest.cpp
                    utils/StringManipulationUTest.cpp
	                ConnectionUTest.cpp
	                utils/PartitionsUTest.cpp
	                OptionsUTest.cpp
	                TaskUTest.cpp
	                ObjectPoolUTest.cpp
	                )

set(UTEST_HDR_FILES utils/DeferUTest.hpp
	                utils/FlagParserUTest.hpp
	                utils/StringManipulationUTest.hpp
	                utils/PartitionsUTest.hpp
	                ConnectionUTest.hpp
	                OptionsUTest.hpp
	                TaskUTest.hpp
	                ObjectPoolUTest.hpp
	                )

if(EGrabber_FOUND)
	list(APPEND SRC_FILES EuresysFrameGrabber.cpp)
	list(APPEND HDR_FILES EuresysFrameGrabber.hpp)
endif(EGrabber_FOUND)


#----------------------------libtorch------------------------------------
#list(APPEND CMAKE_PREFIX_PATH "libtorch")
#find_package(Torch REQUIRED)
#include_directories(${TORCH_INCLUDE_DIRS})
#----------------------------libtorch------------------------------------

#----------------------------libtorch-cxx11-abi------------------------------------
#list(APPEND CMAKE_PREFIX_PATH "libtorch-cxx11-abi")
#find_package(Torch REQUIRED)
#include_directories(${TORCH_INCLUDE_DIRS})
#message("TORCH_INCLUDE_DIRS ---> " ${TORCH_INCLUDE_DIRS})
#message("TORCH_LIBRARIES ---> " ${TORCH_LIBRARIES})
#----------------------------libtorch-cxx11-abi------------------------------------

#----------------------------libtorch-cxx11-abi------------------------------------
#include_directories("libtorch-cxx11-abi/lib.linux-x86_64-3.7/torch/lib/include")
#include_directories("libtorch-cxx11-abi/lib.linux-x86_64-3.7/torch/lib/include/torch/csrc/api/include")
#link_directories("/home/serhii/artemis/src/libtorch-cxx11-abi/lib")
#link_directories("/home/serhii/artemis/src/libtorch-cxx11-abi/lib.linux-x86_64-3.7/torch/lib")
#----------------------------libtorch-cxx11-abi------------------------------------

#----------------------------libtorch-test-----------------------------------
list(APPEND CMAKE_PREFIX_PATH "libtorch")
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
#----------------------------libtorch-test-----------------------------------


add_library(artemis-common STATIC ${SRC_FILES} ${HDR_FILES})
target_link_libraries(artemis-common ${GLOG_LIBRARIES}
	                                 ${EGRABBER_LIBRARIES}
									 fort-time
	                                 fort-tags
	                                 fort-hermes-cpp
	                                 ${PROTOBUF_LIBRARIES}
	                                 ${OpenCV_LIBS}
	                                 ${TBB_LIBRARIES}
	                                 glfw
	                                 ${GLEW_LIBRARIES}
	                                 OpenGL::GL
	                                 freetype-gl
	                                 ${FONTCONFIG_LIBRARIES}
	                                 boost_system
	                                 "-ldl"
									#   ${ONNX_RUNTIME_LIB}
									)

add_library(Trophallaxis SHARED detectors/Trophallaxis.cpp)
set_target_properties(Trophallaxis PROPERTIES
					LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/lib)
add_library(Tracking SHARED detectors/Tracking.cpp)
set_target_properties(Tracking PROPERTIES
					LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/lib)

add_executable(artemis main.cpp)
target_link_libraries(artemis artemis-common Trophallaxis ${TORCH_LIBRARIES})
target_link_libraries(artemis artemis-common Tracking ${TORCH_LIBRARIES})

add_check_test(NAME artemis
	           FILES ${UTEST_SRC_FILES} ${UTEST_HDR_FILES}
	           INCLUDE_DIRS ${GMOCK_INCLUDE_DIRS}
	           LIBRARIES gmock artemis-common)

install(TARGETS artemis DESTINATION bin)
