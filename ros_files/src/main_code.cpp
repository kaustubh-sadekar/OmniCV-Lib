#include<iostream>
#include<opencv2/opencv.hpp>
#include"utils.hpp"
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
  //#######  1. Example for fisheye2eqrect  #############

  frame = cv::imread("./data/fisheye_horizontal2.jpg");

  cv::imshow(WINDOW_NAME,frame);
  cv::waitKey(0);

  fisheyeImgConv mapper1("./fisheyeParams.txt");

  mapper1.fisheye2equirect(frame, outFrame, cv::Size (800,400));
  cv::imshow("output",outFrame);
  cv::waitKey(0);
  mapper1.applyMap(0,frame,outFrame);
  cv::imshow("output 2",outFrame);
  cv::waitKey(0);
  */

  /*
  //######  2. Example for eqrect2cubemap  ############
  frame = cv::imread("./data/equirect2.jpeg");
  fisheyeImgConv mapper1;

  cv::imshow(WINDOW_NAME,frame);
  cv::waitKey(0);

  mapper1.eqrect2cubemap(frame,outFrame,256,true,true);

  cv::imshow(WINDOW_NAME,outFrame);
  cv::waitKey(0);

  mapper1.applyMap(1,frame,outFrame);
  cv::imshow(WINDOW_NAME,outFrame);
  cv::waitKey(0);
  */

  /*
  // ########  3. Example for cubemap to equirectangular image  ########
  frame = cv::imread("../data/cubemap_dice.jpg");
  cv::imshow(WINDOW_NAME,frame);
  cv::waitKey(0);

  fisheyeImgConv mapper1;
  mapper1.cubemap2equirect(frame,cv::Size (800,400),outFrame);
  cv::imshow(WINDOW_NAME,outFrame);
  cv::waitKey(0);
  */

  /*
  // ####### 4. Example for equirectangular to perspective image conversion  #############
  frame = cv::imread("../data/equirect_temp1.jpg");
  cv::imshow(WINDOW_NAME,frame);
  cv::waitKey(0);
  fisheyeImgConv mapper1;
  mapper1.equirect2persp(frame,outFrame,90,120,45, 400,400);
  */


  /* //###### 5. Example for cubemap to perspective image conversion  ##################
  frame = cv::imread("../data/cubemap_dice.jpg");
  cv::imshow(WINDOW_NAME,frame);
  cv::waitKey(0);
  fisheyeImgConv mapper1;
  mapper1.cubemap2persp(frame,outFrame,90,120,45, 400,400);
  */


  /*// ##### 6. Example for equirectangular to fisheye image conversion using Unified camera model ##########
  frame = cv::imread("../data/equirect_temp1.jpg");
  cv::imshow(WINDOW_NAME,frame);
  cv::waitKey(0);
  fisheyeImgConv mapper1;
  mapper1.equirect2Fisheye_UCM(frame,outFrame, cv::Size (250,250),100,0.9);
  // mapper1.equirect2Fisheye_UCM(frame,outFrame, cv::Size (250,250),50,0.5);
  */

  /*// ##### 7. Example for equirectangular to fisheye image conversion using Extended Unified camera model ##########
  frame = cv::imread("../data/equirect_temp1.jpg");
  cv::imshow(WINDOW_NAME,frame);
  cv::waitKey(0);
  fisheyeImgConv mapper1;
  mapper1.equirect2Fisheye_EUCM(frame,outFrame, cv::Size (250,250),100,0.4,2);
  */

  /*
  // ##### 8. Example for equirectangular to fisheye image conversion using Field of View camera model ##########
  frame = cv::imread("../data/equirect_temp1.jpg");
  cv::imshow(WINDOW_NAME,frame);
  cv::waitKey(0);
  fisheyeImgConv mapper1;
  mapper1.equirect2Fisheye_FOV(frame,outFrame, cv::Size (250,250),50,1.2);
  */

  // /*
  // ##### 9. Example for equirectangular to fisheye image conversion using Double Sphere camera model ##########
  frame = cv::imread("equirect_temp1.jpg");
  cv::imshow(WINDOW_NAME,frame);
  cv::waitKey(0);
  fisheyeImgConv mapper1;
  mapper1.equirect2Fisheye_DS(frame,outFrame, cv::Size (250,250),50,0.4,0.8);
  cv::imshow("output",outFrame);
  cv::waitKey(0);
  // */


  return 0;
}