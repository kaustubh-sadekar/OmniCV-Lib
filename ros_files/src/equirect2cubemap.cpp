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

class equirect2cubemap
{
  int cube_side;
  bool modif;
  bool dice;

  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  fisheyeImgConv mapper1;

  std::string file_path;

public:
  equirect2cubemap(int cube_side=256, bool modif=0, bool dice=0)
    : it_(nh_),cube_side{cube_side},modif{modif},dice{dice}
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/omnicv/equirect", 1,
      &equirect2cubemap::imageCb, this);
    image_pub_ = it_.advertise("/omnicv/cubemap", 1);
    cv::namedWindow(OPENCV_WINDOW);

  }

  ~equirect2cubemap()
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

    // mapper1.equirect2cubemap(cv_ptr->image,temp1, cv::Size (this->outW,this->outH));
    mapper1.equirect2cubemap(cv_ptr->image,temp1,this->cube_side,this->modif,this->dice);
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
  int cube_side{256};
  bool modif{0};
  bool dice{0};

  if (argc == 2)
  {
    cube_side = std::stoi(argv[1]);
  }

  if (argc == 3)
  {
    cube_side = std::stoi(argv[1]);
    modif = std::stoi(argv[2]);
  }

  if (argc == 4)
  {
    cube_side = std::stoi(argv[1]);
    modif = std::stoi(argv[2]);
    dice = std::stoi(argv[3]);
  }
  

  ros::init(argc, argv, "omnicv_equirect2cubemap");
  equirect2cubemap ic(cube_side,modif,dice);
  ros::spin();
  return 0;
}