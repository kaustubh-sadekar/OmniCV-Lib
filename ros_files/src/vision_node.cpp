#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

int main(int argc, char** argv)
{
  // Initialize the node
  ros::init(argc, argv, "vision_node");

  // Defining the nodehandle
  ros::NodeHandle nh;

  // Initialize ImageTransport instance with the above defined node handle
  image_transport::ImageTransport it(nh);

  // Creating ImageTransport publisher
  image_transport::Publisher pub = it.advertise("/omnicv/equirect", 1);
  
  // Reading an equirectangular image
  cv::Mat image = cv::imread("equirect.jpg", CV_LOAD_IMAGE_COLOR);
  cv::waitKey(30);
  
  // Converting the opencv image to ros format image using cv_bridge
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();

  ros::Rate loop_rate(5);
  while (nh.ok()) 
  {
    cv::imshow("Publishing image",image);
    cv::waitKey(1);
    pub.publish(msg);
    ros::spinOnce();
    loop_rate.sleep();
  }
}