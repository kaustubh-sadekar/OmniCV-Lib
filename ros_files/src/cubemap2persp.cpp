#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Float32.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils.hpp"

static const std::string OPENCV_WINDOW = "Image window";

class cubemap2persp
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  fisheyeImgConv mapper1;
  float FOV{90},theta{0},phi{0};

  ros::Subscriber sub_FOV = nh_.subscribe("/omnicv/cubemap2persp/FOV",1,&cubemap2persp::FOVCb,this);
  ros::Subscriber sub_theta = nh_.subscribe("/omnicv/cubemap2persp/theta",1,&cubemap2persp::thetaCb,this);
  ros::Subscriber sub_phi = nh_.subscribe("/omnicv/cubemap2persp/phi",1,&cubemap2persp::phiCb,this);
  
  int outH,outW;

public:
  cubemap2persp(int outH, int outW)
    : it_(nh_),outH{outH},outW{outW}
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/omnicv/cubemap", 1,
      &cubemap2persp::imageCb, this);
    image_pub_ = it_.advertise("/omnicv/persp", 1);
    cv::namedWindow(OPENCV_WINDOW);

    this->outH = outH;
    this->outW = outW;
  }

  ~cubemap2persp()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void FOVCb(const std_msgs::Float32::ConstPtr& msg)
  {
    this->FOV = msg->data;
  }

  void thetaCb(const std_msgs::Float32::ConstPtr& msg)
  {
    this->theta = msg->data;
  }

  void phiCb(const std_msgs::Float32::ConstPtr& msg)
  {
    this->phi = msg->data;
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv::Mat temp1;
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // // Draw an example circle on the video stream
    // if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
    //   cv::circle(cv_ptr->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));
    mapper1.cubemap2persp(cv_ptr->image,temp1,this->FOV,this->theta,this->phi,this->outH,this->outW);
    
    temp1.copyTo(cv_ptr->image);


    // Update GUI Window
    cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    cv::waitKey(3);

    // Output modified video stream
    image_pub_.publish(cv_ptr->toImageMsg());

    temp1.release();
  }
};

int main(int argc, char *argv[])
{
  int outH,outW;
  
  if (argc>1)
  {
  	outH = std::atoi(argv[1]);
  	outW = std::atoi(argv[2]);
  }
  else
  {
  	outH = 400; // Default output image height
  	outW = 400; // Default output image width
  }

  ros::init(argc, argv, "omnicv_cubemap2persp");
  cubemap2persp ic(outH,outW);
  ros::spin();
  return 0;
}