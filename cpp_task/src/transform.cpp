#include "transform.hpp"
void imageTransform(cv::Mat &img)
{
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
}
