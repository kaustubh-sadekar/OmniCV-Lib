#include <ros/ros.h>
#include<ros/package.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Float32.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils.hpp"

static const std::string OPENCV_WINDOW = "Image window";
// static const std::string file_path = ros::package::getPath("omnicv");

class cubemap2equirect
{
  int outH,outW;
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  fisheyeImgConv mapper1;

  std::string file_path;

public:
  cubemap2equirect(int outW=800, int outH=400)
    : it_(nh_),outW{outW},outH{outH}
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/omnicv/cubemap", 1,
      &cubemap2equirect::imageCb, this);
    image_pub_ = it_.advertise("/omnicv/equirect", 1);
    cv::namedWindow(OPENCV_WINDOW);

  }

  ~cubemap2equirect()
  {
    cv::destroyWindow(OPENCV_WINDOW);
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

    mapper1.cubemap2equirect(cv_ptr->image,cv::Size (this->outW,this->outH),temp1);
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
  int outW{800},outH{400};

  if (argc == 3)
  {
    outW = std::stoi(argv[1]);
    outH = std::stoi(argv[2]);
  }


  ros::init(argc, argv, "omnicv_equirect2cubemap");
  cubemap2equirect ic(outW,outH);
  ros::spin();
  return 0;
}