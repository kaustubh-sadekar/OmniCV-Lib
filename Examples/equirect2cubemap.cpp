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
  fisheyeImgConv mapper1;

  cv::imshow(WINDOW_NAME,frame);
  cv::waitKey(0);

  mapper1.eqrect2cubemap(frame,outFrame,256,true,true);

  cv::imshow(WINDOW_NAME,outFrame);
  cv::waitKey(0);

  mapper1.applyMap(1,frame,outFrame);
  cv::imshow(WINDOW_NAME,outFrame);
  cv::waitKey(0);
  
  return 0;
}