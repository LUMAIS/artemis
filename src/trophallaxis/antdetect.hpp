#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

template <typename T> T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */

template <typename T> std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

/**
 * @brief Print ONNX tensor data type
 * https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L93
 * @param os
 * @param type
 * @return std::ostream&
 */
std::ostream& operator<<(std::ostream& os, const ONNXTensorElementDataType& type)
{
    switch (type)
    {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            os << "undefined";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            os << "float";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            os << "uint8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            os << "int8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            os << "uint16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            os << "int16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            os << "int32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            os << "int64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            os << "std::string";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            os << "bool";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            os << "float16";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            os << "double";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            os << "uint32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            os << "uint64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            os << "float real + float imaginary";
            break;
        case ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            os << "double real + float imaginary";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            os << "bfloat16";
            break;
        default:
            break;
    }

    return os;
}

std::vector<std::string> readLabels(std::string& labelFilepath)
{
    std::vector<std::string> labels;
    std::string line;
    std::ifstream fp(labelFilepath);
    while (std::getline(fp, line))
    {
        labels.push_back(line);
    }
    return labels;
}

float detect(cv::Mat imageBGR, std::vector<std::string> labels, Ort::Session &session)
{

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    //std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    //std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;

    const char* inputName = session.GetInputName(0, allocator);
    //std::cout << "Input Name: " << inputName << std::endl;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
   //std::cout << "Input Type: " << inputType << std::endl;

    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    //std::cout << "Input Dimensions: " << inputDims << std::endl;

    const char* outputName = session.GetOutputName(0, allocator);
    //std::cout << "Output Name: " << outputName << std::endl;

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    //std::cout << "Output Type: " << outputType << std::endl;

    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    //std::cout << "Output Dimensions: " << outputDims << std::endl;

    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;

    cv::resize(imageBGR, resizedImageBGR,cv::Size(inputDims.at(2), inputDims.at(3)),cv::InterpolationFlags::INTER_CUBIC);
    cv::cvtColor(resizedImageBGR, resizedImageRGB,cv::ColorConversionCodes::COLOR_BGR2RGB);
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

    cv::Mat channels[3];
    cv::split(resizedImage, channels);

    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;
    cv::merge(channels, 3, resizedImage);
    // HWC to CHW

    cv::dnn::blobFromImage(resizedImage, preprocessedImage);

    size_t inputTensorSize = vectorProduct(inputDims);
    std::vector<float> inputTensorValues(inputTensorSize);
    inputTensorValues.assign(preprocessedImage.begin<float>(),
                             preprocessedImage.end<float>());

    size_t outputTensorSize = vectorProduct(outputDims);
    assert(("Output tensor size should equal to the label set size.",
            labels.size() == outputTensorSize));
    std::vector<float> outputTensorValues(outputTensorSize);

    std::vector<const char*> inputNames{inputName};
    std::vector<const char*> outputNames{outputName};
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),inputDims.size()));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues.data(), outputTensorSize,outputDims.data(), outputDims.size()));

    session.Run(Ort::RunOptions{nullptr}, inputNames.data(),inputTensors.data(), 1, outputNames.data(),outputTensors.data(), 1);

    int predId = 0;
    float activation = 0;
    float maxActivation = std::numeric_limits<float>::lowest();
    float expSum = 0;
    for (int i = 0; i < labels.size(); i++)
    {
        activation = outputTensorValues.at(i);
        expSum += std::exp(activation);
        if (activation > maxActivation)
        {
            predId = i;
            maxActivation = activation;
        }
    }
    
    if(exp(maxActivation) / expSum > 0.1)
    {
        std::cout << "Predicted Label ID: " << predId << std::endl;
        std::cout << "Predicted Label: " << labels.at(predId) << std::endl;
        std::cout << "Uncalibrated Confidence: " << std::exp(maxActivation) / expSum<< std::endl<< std::endl;

        if(predId == 92 || predId == 29 || predId == 94)
        return 1.0;
    }
    else
    {
        if(predId == 92 || predId == 29 || predId == 94)
            return 0.7;
    }

    return exp(maxActivation) / expSum;
}

void drawrec(cv::Mat &image, cv::Point2f p, float d, int koef)
{
	cv::Point2f p1;
    cv::Point2f p2;
    cv::Point2f p3;
    cv::Point2f p4;

	p1.x = p.x/koef;
	p1.y = p.y/koef;
	
    p2.x = p1.x+100/koef;
    p2.y = p1.y;

    p3.x = p2.x;
    p3.y = p2.y+100/koef;

    p4.x = p3.x-100/koef;
    p4.y = p3.y;

    u_char R, G, B;

    if(d <= 0.3)
    {
        R = 40;
        G = 160;
        B = 0;
    }
    else if(d <= 0.8)
    {
        R = 255;
        G = 128;
        B = 0;
    }
    else
    {
        R = 255;
        G = 0;
        B = 0;
    }
    
    cv::line(image, p1, p2, cv::Scalar(B, G, R), 2, 8, 0);
    cv::line(image, p2, p3, cv::Scalar(B, G, R), 2, 8, 0);
    cv::line(image, p3, p4, cv::Scalar(B, G, R), 2, 8, 0);
    cv::line(image, p4, p1, cv::Scalar(B, G, R), 2, 8, 0);
}

std::vector<std::array<float,2>> TDetect(bool useCUDA, std::string modelFilepath, std::string labelFilepath, cv::Mat imageBGR, size_t nThreads);