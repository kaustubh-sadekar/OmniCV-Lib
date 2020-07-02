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
  // cv::resizeWindow(WINDOW_NAME, 400, 400);

  cv::Mat frame;
  cv::Mat outFrame;

  /*
  // ##### 1. Example for equirectangular to fisheye image conversion using Unified camera model ##########
  frame = cv::imread("../data/equirect_temp1.jpg");
  cv::imshow(WINDOW_NAME,frame);
  cv::waitKey(0);
  fisheyeImgConv mapper1;
  mapper1.equirect2Fisheye_UCM(frame,outFrame, cv::Size (250,250),100,0.9);
  cv::imshow("Unified Camera Model model output",outFrame);
  cv::waitKey(0);
  */

  /*
  // ##### 2. Example for equirectangular to fisheye image conversion using Extended Unified camera model ##########
  frame = cv::imread("../data/equirect_temp1.jpg");
  cv::imshow(WINDOW_NAME,frame);
  cv::waitKey(0);
  fisheyeImgConv mapper1;
  mapper1.equirect2Fisheye_EUCM(frame,outFrame, cv::Size (250,250),100,0.4,2);
  cv::imshow("Extended Unified Camera Model model output",outFrame);
  cv::waitKey(0);
  */

  /*
  // ##### 3. Example for equirectangular to fisheye image conversion using Field of View camera model ##########
  frame = cv::imread("../data/equirect_temp1.jpg");
  cv::imshow(WINDOW_NAME,frame);
  cv::waitKey(0);
  fisheyeImgConv mapper1;
  mapper1.equirect2Fisheye_FOV(frame,outFrame, cv::Size (250,250),50,1.2);
  cv::imshow("Field of View model output",outFrame);
  cv::waitKey(0);
  */

  // /*
  // ##### 4. Example for equirectangular to fisheye image conversion using Double Sphere camera model ##########
  frame = cv::imread("../data/equirect_temp1.jpg");
  cv::imshow(WINDOW_NAME,frame);
  cv::waitKey(0);
  fisheyeImgConv mapper1;
  mapper1.equirect2Fisheye_DS(frame,outFrame, cv::Size (250,250),50,0.4,0.8);
  cv::imshow("Double Sphere model output",outFrame);
  cv::waitKey(0);
  // */
  return 0;
}