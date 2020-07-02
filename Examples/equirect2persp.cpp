#include<iostream>
#include<opencv2/opencv.hpp>
#include"../utils.hpp"
#include <opencv2/core/core.hpp>

// Creating the display window
int H = 500;
int W = 500;

std::string WINDOW_NAME{"viewer"};

int main()
{
  cv::namedWindow(WINDOW_NAME,CV_WINDOW_NORMAL);

  cv::Mat frame;
  cv::Mat outFrame;

  frame = cv::imread("../data/equirect_temp1.jpg");
  cv::imshow(WINDOW_NAME,frame);
  cv::waitKey(0);
  fisheyeImgConv mapper1;
  mapper1.equirect2persp(frame,outFrame,90,120,45, 400,400);

  cv::imshow("persp",outFrame);
  cv::waitKey(0);
  
  return 0;
}