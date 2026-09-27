#include <iostream>
#include <opencv2/opencv.hpp>
#include "transform.hpp"

int main(int argc, char** argv)
{
    if(argc !=3)
    {
        std::cout << "Usage: ./program input.mp4 output.mp4" << std::endl;
        return -1;
    }
    cv::VideoCapture cap(argv[1]);
    if(!cap.isOpened())
    {
        std::cout << "cannot open video" << std::endl;
        return -1;
    }
    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::Size size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter writer(argv[2], cv::VideoWriter::fourcc('m','p','4','v'), fps, size);

    cv::Mat frame;
    while(cap.read(frame))
    {
        imageTransform(frame);
        writer.write(frame);
    }
    cap.release();
    writer.release();
    std::cout << "Video finished!" << std::endl;
    return 0;
}
