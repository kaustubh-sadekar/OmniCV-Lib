#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Float32.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils.hpp"

static const std::string OPENCV_WINDOW = "Image window";

class equirect2fisheye
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  fisheyeImgConv mapper1;
  float alpha{0},beta{0},gamma{0},f{50},xi{0.5},a_{1},b_{1},w_{1};
  
  ros::Subscriber sub_alpha = nh_.subscribe("/omnicv/equirect2fisheye/alpha",1,&equirect2fisheye::alphaCb,this);
  ros::Subscriber sub_beta = nh_.subscribe("/omnicv/equirect2fisheye/beta",1,&equirect2fisheye::betaCb,this);
  ros::Subscriber sub_gamma = nh_.subscribe("/omnicv/equirect2fisheye/gamma",1,&equirect2fisheye::gammaCb,this);
  ros::Subscriber sub_f = nh_.subscribe("/omnicv/equirect2fisheye/f",1,&equirect2fisheye::fCb,this);
  ros::Subscriber sub_xi = nh_.subscribe("/omnicv/equirect2fisheye/xi",1,&equirect2fisheye::xiCb,this);
  ros::Subscriber sub_a_ = nh_.subscribe("/omnicv/equirect2fisheye/a_",1,&equirect2fisheye::aCb,this);
  ros::Subscriber sub_b_ = nh_.subscribe("/omnicv/equirect2fisheye/b_",1,&equirect2fisheye::bCb,this);
  ros::Subscriber sub_w_ = nh_.subscribe("/omnicv/equirect2fisheye/w_",1,&equirect2fisheye::wCb,this);
  
  int mode; // 0-UCM, 1-EUCM, 2-FOV, 3-DS
  int outH,outW;

public:
  equirect2fisheye(int mode, int outH, int outW)
    : it_(nh_),mode{mode},outH{outH},outW{outW}
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/omnicv/equirect", 1,
      &equirect2fisheye::imageCb, this);
    image_pub_ = it_.advertise("/omnicv/fisheye", 1);
    cv::namedWindow(OPENCV_WINDOW);

    this->mode = mode;
    this->outH = outH;
    this->outW = outW;
  }

  ~equirect2fisheye()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void alphaCb(const std_msgs::Float32::ConstPtr& msg)
  {
    this->alpha = msg->data;
  }

  void betaCb(const std_msgs::Float32::ConstPtr& msg)
  {
    this->beta = msg->data;
  }

  void gammaCb(const std_msgs::Float32::ConstPtr& msg)
  {
    this->gamma = msg->data;
  }

  void fCb(const std_msgs::Float32::ConstPtr& msg)
  {
    this->f = msg->data;
  }

  void xiCb(const std_msgs::Float32::ConstPtr& msg)
  {
    this->xi = msg->data;
  }

  void aCb(const std_msgs::Float32::ConstPtr& msg)
  {
    this->a_ = msg->data;
  }

  void bCb(const std_msgs::Float32::ConstPtr& msg)
  {
    this->b_ = msg->data;
  }

  void wCb(const std_msgs::Float32::ConstPtr& msg)
  {
    this->w_ = msg->data;
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
    if (this->mode == 0)
	    mapper1.equirect2Fisheye_UCM(cv_ptr->image,temp1, cv::Size (this->outH,this->outW),this->f,this->xi,this->alpha,this->beta,this->gamma);
	else if (this->mode == 1)
		mapper1.equirect2Fisheye_EUCM(cv_ptr->image,temp1, cv::Size (this->outH,this->outW),this->f,this->a_,this->b_,this->alpha,this->beta,this->gamma);
	else if (this->mode == 2)
		mapper1.equirect2Fisheye_FOV(cv_ptr->image,temp1, cv::Size (this->outH,this->outW),this->f,this->w_,this->alpha,this->beta,this->gamma);
	else if (this->mode == 3)
		mapper1.equirect2Fisheye_DS(cv_ptr->image,temp1, cv::Size (this->outH,this->outW),this->f,this->a_,this->xi,this->alpha,this->beta,this->gamma);
	else
	{
		std::cout << "Wrong mode entered .... " << this->mode << std::endl;
		std::exit(-1);
	}

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
  int mode = std::atoi(argv[1]);
  int outH,outW;
  
  if (argc>2)
  {
  	outH = std::atoi(argv[2]);
  	outW = std::atoi(argv[3]);
  }
  else
  {
  	outH = 400; // Default output image height
  	outW = 400; // Default output image width
  }

  ros::init(argc, argv, "omnicv_equirect2fisheye");
  equirect2fisheye ic(mode,outH,outW);
  ros::spin();
  return 0;
}