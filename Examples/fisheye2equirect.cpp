#include<iostream>
#include<opencv2/opencv.hpp>
#include"../omnicv/utils.hpp"
#include <opencv2/core/core.hpp>
#define CV_WINDOW_NORMAL 0

// Creating the display window
int H = 500;
int W = 500;

std::string WINDOW_NAME{"viewer"};

int main()
{
  cv::namedWindow(WINDOW_NAME,CV_WINDOW_NORMAL);

  cv::Mat frame;
  cv::Mat outFrame;

  frame = cv::imread("../data/fisheye_horizontal.png");

  cv::imshow(WINDOW_NAME,frame);
  cv::waitKey(0);

  fisheyeImgConv mapper1("../fisheyeParams.txt");

  mapper1.fisheye2equirect(frame, outFrame, cv::Size (800,400));
  cv::imshow("output",outFrame);
  cv::waitKey(0);

  return 0;
}
