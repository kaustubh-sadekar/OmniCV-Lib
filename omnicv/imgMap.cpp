#include"utils.hpp"
#include<fstream>
#include<iostream>
#include<string>
#include <complex> 
#define CV_INTER_CUBIC 2

fisheyeImgConv::fisheyeImgConv(std::string paramFilePath)
: filePath {paramFilePath} {}

void fisheyeImgConv::meshgrid(const cv::Range &xgv, const cv::Range &ygv,cv::Mat &X, cv::Mat &Y)
{
  std::vector<int> t_x, t_y;
  for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
  for (int i = ygv.start; i <= ygv.end; i++) t_y.push_back(i);

  cv::repeat(cv::Mat(t_x).reshape(1,1), cv::Mat(t_y).total(), 1, X);
  cv::repeat(cv::Mat(t_y).reshape(1,1).t(), 1, cv::Mat(t_x).total(), Y);
}

cv::Mat fisheyeImgConv::RMat(double alpha, double beta, double gamma)
{	
	cv::Mat R;

	cv::Mat Rx = (cv::Mat_<double>(3,3) << 1,0,0,0,std::cos(alpha),-1*std::sin(alpha),0,std::sin(alpha),std::cos(alpha));
	cv::Mat Ry = (cv::Mat_<double>(3,3) << std::cos(beta),0,1*std::sin(beta),0,1,0,-1*std::sin(beta),0,std::cos(beta));
	cv::Mat Rz = (cv::Mat_<double>(3,3) << std::cos(gamma),-1*std::sin(gamma),0,std::sin(gamma),std::cos(gamma),0,0,0,1);
	R = Rz*Ry*Rx;

	return R;
}
void fisheyeImgConv::fisheye2equirect(const cv::Mat &srcFrame, cv::Mat &dstFrame, cv::Size outShape, int aperture, int delx, int dely, int radius, bool edit_mode)
{
	this->Hs = srcFrame.rows;
	this->Ws = srcFrame.cols;
	this->Hd = outShape.height;
	this->Wd = outShape.width;

	this->map_x = cv::Mat::zeros(this->Hd,this->Wd,CV_32FC1);
	this->map_y = cv::Mat::zeros(this->Hd,this->Wd,CV_32FC1);

	this->Cx = int(this->Ws/2 - delx);
	this->Cy = int(this->Hs/2 - dely);


	if (!(edit_mode))
	{
		std::fstream f;
		std::string val;
		int temp;

		f.open(this->filePath,ios::in);
		if(!f)
			std::cout << "File path wrong! ";
		else
		{
			std::getline(f, val);
			this->radius = stoi(val);
			std::getline(f, val);
			this->aperture = stoi(val);
			std::getline(f, val);
			delx = stoi(val);
			std::getline(f, val);
			dely = stoi(val);
		}
	}

	float x,y,z,r,theta;

	for(size_t i{0}; i<this->Hd; i++)
	{
		float* dataX = this->map_x.ptr<float>(i);
		float* datay = this->map_y.ptr<float>(i);

		for(size_t j{0}; j<this->Wd; j++)
		{
			x = this->radius*std::cos((i*1.0/this->Hd-0.5)*CV_PI)*std::cos((j*1.0/this->Hd-0.5)*CV_PI);
			y = this->radius*std::cos((i*1.0/this->Hd-0.5)*CV_PI)*std::sin((j*1.0/this->Hd-0.5)*CV_PI);
			z = this->radius*std::sin((i*1.0/this->Hd-0.5)*CV_PI);

			r = 2*std::atan2(std::sqrt(x*x + z*z),y)/CV_PI*180/this->aperture*this->radius;
			theta = std::atan2(z,x);

			dataX[j] = float(r*std::cos(theta)+this->Cx);
			datay[j] = float(r*std::sin(theta)+this->Cy);
		}
	}

	cv::remap(srcFrame, dstFrame, this->map_x, this->map_y, CV_INTER_CUBIC);
}

void fisheyeImgConv::applyMap(const int &map, const cv::Mat &srcFrame, cv::Mat &dstFrame)
{
	if(map == 0)
		cv::remap(srcFrame, dstFrame, this->map_x, this->map_y, CV_INTER_CUBIC);
	else if (map == 1)
	{
		cv::remap(srcFrame, dstFrame, this->map_x, this->map_y, CV_INTER_CUBIC);

		if(this->dice)
		{
			cv::Mat temp,t_;
			temp = cv::Mat::zeros(this->side*3, this->side*4, CV_8UC3);

			dstFrame.rowRange(0,this->side).colRange(0,this->side).copyTo(temp.rowRange(this->side,this->side*2).colRange(this->side,this->side*2));
			dstFrame.rowRange(0,this->side).colRange(this->side*3,this->side*4).copyTo(temp.rowRange(this->side,this->side*2).colRange(0,this->side));
			dstFrame.rowRange(0,this->side).colRange(this->side*5,this->side*6).copyTo(temp.rowRange(this->side*2,this->side*3).colRange(this->side,this->side*2));
			cv::flip(dstFrame.rowRange(0,this->side).colRange(this->side*4,this->side*5), t_,0);
			t_.copyTo(temp.rowRange(0,this->side).colRange(this->side,this->side*2));
			cv::flip(dstFrame.rowRange(0,this->side).colRange(this->side,this->side*2), t_,1);
			t_.copyTo(temp.rowRange(this->side,this->side*2).colRange(this->side*2,this->side*3));
			cv::flip(dstFrame.rowRange(0,this->side).colRange(this->side*2,this->side*3), t_,1);
			t_.copyTo(temp.rowRange(this->side,this->side*2).colRange(this->side*3,this->side*4));

			dstFrame = temp;
			temp.release();
			t_.release();
		}
	}
	else
	{
		std::cout << "Wrong map type provided !";
		exit(-1);
	}
}

void fisheyeImgConv::equirect2cubemap(const cv::Mat &srcFrame, cv::Mat &dstFrame, const int side, const bool modif, const bool dice)
{
	this->dice = dice;
	this->side = side;

	this->map_x = cv::Mat::zeros(this->side, this->side*6, CV_32FC1);
	this->map_y = cv::Mat::zeros(this->side, this->side*6, CV_32FC1);

	
	float x,y,z;
	float phi,theta;

	if(!(modif))
	{
		for(size_t i{0}; i<this->map_x.rows; i++)
		{
			for(size_t j{0}; j<this->map_x.cols; j++)
			{
				if (j/this->side == 0)
				{
					x = ((j%this->side)*1.0/this->side) - 0.5;
					y = -((i*1.0/this->side) - 0.5);
					z = 0.5;

					phi = std::atan2(x,z);
					theta = std::atan2(y,std::sqrt(x*x+z*z));

					this->map_x.at<float>(i,j) = (phi/(2*CV_PI) + 0.5)*srcFrame.cols;
					this->map_y.at<float>(i,j) = (-theta/CV_PI + 0.5)*srcFrame.rows;
				}
				else if(j/this->side == 1)
				{
					z = ((j%this->side)*1.0/this->side) - 0.5;
					y = -((i*1.0/this->side) - 0.5);
					x = 0.5;

					phi = std::atan2(x,z);
					theta = std::atan2(y,std::sqrt(x*x+z*z));

					this->map_x.at<float>(i,j) = (phi/(2*CV_PI) + 0.5)*srcFrame.cols;
					this->map_y.at<float>(i,j) = (-theta/CV_PI + 0.5)*srcFrame.rows;
				}
				else if(j/this->side == 2)
				{
					x = ((j%this->side)*1.0/this->side) - 0.5;
					y = -((i*1.0/this->side) - 0.5);
					z = -0.5;

					phi = std::atan2(x,z);
					theta = std::atan2(y,std::sqrt(x*x+z*z));

					this->map_x.at<float>(i,j) = (phi/(2*CV_PI) + 0.5)*srcFrame.cols;
					this->map_y.at<float>(i,j) = (-theta/CV_PI + 0.5)*srcFrame.rows;
				}
				else if(j/this->side == 3)
				{
					z = ((j%this->side)*1.0/this->side) - 0.5;
					y = -((i*1.0/this->side) - 0.5);
					x = -0.5;

					phi = std::atan2(x,z);
					theta = std::atan2(y,std::sqrt(x*x+z*z));

					this->map_x.at<float>(i,j) = (phi/(2*CV_PI) + 0.5)*srcFrame.cols;
					this->map_y.at<float>(i,j) = (-theta/CV_PI + 0.5)*srcFrame.rows;
				}
				else if(j/this->side == 4)
				{
					x = ((j%this->side)*1.0/this->side) - 0.5;
					z = -((i*1.0/this->side) - 0.5);
					y = 0.5;

					phi = std::atan2(x,z);
					theta = std::atan2(y,std::sqrt(x*x+z*z));

					this->map_x.at<float>(i,j) = (phi/(2*CV_PI) + 0.5)*srcFrame.cols;
					this->map_y.at<float>(i,j) = (-theta/CV_PI + 0.5)*srcFrame.rows;
				}
				else if(j/this->side == 5)
				{
					x = ((j%this->side)*1.0/this->side) - 0.5;
					z = -((i*1.0/this->side) - 0.5);
					y = -0.5;

					phi = std::atan2(x,z);
					theta = std::atan2(y,std::sqrt(x*x+z*z));

					this->map_x.at<float>(i,j) = (phi/(2*CV_PI) + 0.5)*srcFrame.cols;
					this->map_y.at<float>(i,j) = (-theta/CV_PI + 0.5)*srcFrame.rows;
				}
			}
		}
	}

	else
	{
		for(size_t i{0}; i<this->map_x.rows; i++)
		{
			for(size_t j{0}; j<this->map_x.cols; j++)
			{
				if (j/this->side == 0)
				{
					x = ((j%this->side)*1.0/this->side) - 0.5;
					z = -((i*1.0/this->side) - 0.5);
					y = -0.5;

					phi = std::atan2(x,z);
					theta = std::atan2(y,std::sqrt(x*x+z*z));

					this->map_x.at<float>(i,j) = (phi/(2*CV_PI) + 0.5)*srcFrame.cols;
					this->map_y.at<float>(i,j) = (-theta/CV_PI + 0.5)*srcFrame.rows;
				}
				else if(j/this->side == 1)
				{
					y = -(((j%this->side)*1.0/this->side) - 0.5);
					z = -((i*1.0/this->side) - 0.5);
					x = 0.5;

					phi = std::atan2(x,z);
					theta = std::atan2(y,std::sqrt(x*x+z*z));

					this->map_x.at<float>(i,j) = (phi/(2*CV_PI) + 0.5)*srcFrame.cols;
					this->map_y.at<float>(i,j) = (-theta/CV_PI + 0.5)*srcFrame.rows;
				}
				else if(j/this->side == 2)
				{
					x = ((j%this->side)*1.0/this->side) - 0.5;
					z = -((i*1.0/this->side) - 0.5);
					y = 0.5;

					phi = std::atan2(x,(z));
					theta = std::atan2(y,std::sqrt(x*x+z*z));

					this->map_x.at<float>(i,j) = (phi/(2*CV_PI) + 0.5)*srcFrame.cols;
					this->map_y.at<float>(i,j) = (-theta/CV_PI + 0.5)*srcFrame.rows;
				}
				else if(j/this->side == 3)
				{
					y = -(((j%this->side)*1.0/this->side) - 0.5);
					z = -((i*1.0/this->side) - 0.5);
					x = -0.5;

					phi = std::atan2(x,z);
					theta = std::atan2(y,std::sqrt(x*x+z*z));

					this->map_x.at<float>(i,j) = (phi/(2*CV_PI) + 0.5)*srcFrame.cols;
					this->map_y.at<float>(i,j) = (-theta/CV_PI + 0.5)*srcFrame.rows;
				}
				else if(j/this->side == 4)
				{
					x = ((j%this->side)*1.0/this->side) - 0.5;
					y = ((i*1.0/this->side) - 0.5);
					z = 0.5;

					phi = std::atan2(x,z);
					theta = std::atan2(y,std::sqrt(x*x+z*z));

					this->map_x.at<float>(i,j) = (phi/(2*CV_PI) + 0.5)*srcFrame.cols;
					this->map_y.at<float>(i,j) = (-theta/CV_PI + 0.5)*srcFrame.rows;
				}
				else if(j/this->side == 5)
				{
					x = ((j%this->side)*1.0/this->side) - 0.5;
					y = -((i*1.0/this->side) - 0.5);
					z = -0.5;

					phi = std::atan2(x,z);
					theta = std::atan2(y,std::sqrt(x*x+z*z));

					this->map_x.at<float>(i,j) = (phi/(2*CV_PI) + 0.5)*srcFrame.cols;
					this->map_y.at<float>(i,j) = (-theta/CV_PI + 0.5)*srcFrame.rows;
				}
			}
		}
	}	
	cv::remap(srcFrame, dstFrame, this->map_x, this->map_y, CV_INTER_CUBIC);

	if(this->dice)
	{
		cv::Mat temp,t_;
		temp = cv::Mat::zeros(this->side*3, this->side*4, CV_8UC3);

		dstFrame.rowRange(0,this->side).colRange(0,this->side).copyTo(temp.rowRange(this->side,this->side*2).colRange(this->side,this->side*2));
		dstFrame.rowRange(0,this->side).colRange(this->side*3,this->side*4).copyTo(temp.rowRange(this->side,this->side*2).colRange(0,this->side));
		dstFrame.rowRange(0,this->side).colRange(this->side*5,this->side*6).copyTo(temp.rowRange(this->side*2,this->side*3).colRange(this->side,this->side*2));
		cv::flip(dstFrame.rowRange(0,this->side).colRange(this->side*4,this->side*5), t_,0);
		t_.copyTo(temp.rowRange(0,this->side).colRange(this->side,this->side*2));
		cv::flip(dstFrame.rowRange(0,this->side).colRange(this->side,this->side*2), t_,1);
		t_.copyTo(temp.rowRange(this->side,this->side*2).colRange(this->side*2,this->side*3));
		cv::flip(dstFrame.rowRange(0,this->side).colRange(this->side*2,this->side*3), t_,1);
		t_.copyTo(temp.rowRange(this->side,this->side*2).colRange(this->side*3,this->side*4));

		dstFrame = temp;
		temp.release();
		t_.release();
	}
	
}

void fisheyeImgConv::cubemap2equirect(const cv::Mat &srcFrame1, const cv::Size outShape, cv::Mat &dstFrame)
{
	int h{srcFrame1.rows};
	int w{srcFrame1.cols};

	this->Hs = h;
	this->Ws = w;

	cv::Mat srcFrame;
	
	bool dice{(h*1.0/w) == (3.0/4.0)};
	int side{0};

	if (dice)
	{
		cv::Mat t_;

		side = h/3;

		srcFrame = cv::Mat::zeros(side,side*6,CV_8UC3);

		srcFrame1.rowRange(side,side*2).colRange(side,side*2).copyTo(srcFrame.rowRange(0,side).colRange(0,side));
		srcFrame1.rowRange(side,side*2).colRange(0,side).copyTo(srcFrame.rowRange(0,side).colRange(side*3,side*4));
		srcFrame1.rowRange(side*2,side*3).colRange(side,side*2).copyTo(srcFrame.rowRange(0,side).colRange(side*5,side*6));
		cv::flip(srcFrame1.rowRange(0,side).colRange(side,side*2), t_,0);
		// srcFrame1.rowRange(0,side).colRange(side,side*2).copyTo(t_);
		t_.copyTo(srcFrame.rowRange(0,side).colRange(side*4,side*5));
		// cv::flip(srcFrame1.rowRange(side,side*2).colRange(side*2,side*3), t_,1);
		srcFrame1.rowRange(side,side*2).colRange(side*2,side*3).copyTo(t_);
		t_.copyTo(srcFrame.rowRange(0,side).colRange(side,side*2));
		// cv::flip(srcFrame1.rowRange(side,side*2).colRange(side*3,side*4), t_,1);
		srcFrame1.rowRange(side,side*2).colRange(side*3,side*4).copyTo(t_);
		t_.copyTo(srcFrame.rowRange(0,side).colRange(side*2,side*3));
	}
	else
		srcFrame1.copyTo(srcFrame);

	/*
	NOTE IMP IDEA FOR CODE:
	Keep the cubemap image as it is and do not stack it one over the other. your x coordinate will change as x+ix where i is the face id 
	or the ith face in the cube map. This will be more convinient and easy to impliment. So you find the X,Y as it is and then find the 
	face id and then add the i*x component to the X map. Note that the Y map will be same for all the faces. This will be easier.
	In python we did not do this because we were easily able to index the elements using numpy concepts.
	*/

	int face_w{srcFrame.rows};

	float phi,theta,x_,y_;
	float x,y,z,idx1,idx2;
	int face_id;
	float phi_,theta_;

	this->Hd = outShape.height;
	this->Wd = outShape.width;

	this->map_x = cv::Mat::zeros(this->Hd, this->Wd, CV_32FC1);
	this->map_y = cv::Mat::zeros(this->Hd, this->Wd, CV_32FC1);

	for(size_t j{0}; j< this->map_x.rows; j++)
	{
		for(size_t i{0}; i< this->map_x.cols; i++)
		{
			phi = (i*1.0/this->Wd - 0.5)*2*CV_PI;
			theta = (0.5 - j*1.0/this->Hd)*CV_PI;
			
			if (i < this->Wd/8)
				face_id = 2;
			if ((i >= this->Wd/8)&&(i < this->Wd*3/8))
				face_id = 3;
			if ((i >= this->Wd*3/8)&&(i < this->Wd*5/8))
				face_id = 0;
			if ((i >= this->Wd*5/8)&&(i < this->Wd*7/8))
				face_id = 1;
			if (i >= this->Wd*7/8)
				face_id = 2;

			// phi_ = ((i%int(this->Wd/4))*1.0/(this->Wd/4) - 0.5)*(CV_PI/2);
			phi_ = (((i+int(this->Wd/8))%int(this->Wd/4))*1.0/(this->Wd/4) - 0.5)*(CV_PI/2);

			// break;
			idx1 = this->Hd/2 - (std::atan(std::cos(phi_))*this->Hd/CV_PI);
			idx2 = this->Hd/2 + (std::atan(std::cos(phi_))*this->Hd/CV_PI);

			if (j < idx1)
			{
				x_ = 0.5*std::tan(CV_PI / 2 - theta)*std::sin(phi);
				y_ = -0.5*std::tan(CV_PI / 2 - theta)*std::cos(phi);
				face_id = 4;

				this->map_x.at<float>(j,i) = (x_ + 0.5)*face_w + face_id*face_w -1;
				this->map_y.at<float>(j,i) = (y_ + 0.5)*face_w -1;
			}
			else if (j > idx2)
			{
				x_ = 0.5*std::tan(CV_PI / 2 - std::abs(theta))*std::sin(phi);
				y_ = -0.5*std::tan(CV_PI / 2 - std::abs(theta))*std::cos(phi);
				face_id = 5;
				this->map_x.at<float>(j,i) = (x_ + 0.5)*face_w + face_id*face_w -1 ;
				this->map_y.at<float>(j,i) = (y_ + 0.5)*face_w -1 ;
			}
			else
			{
				x_ = 0.5*std::tan(phi - CV_PI*face_id/2);
				y_ = -0.5*std::tan(theta)/std::cos(phi - CV_PI*face_id/2);
				this->map_x.at<float>(j,i) = (x_ + 0.5)*face_w + face_id*face_w;
				this->map_y.at<float>(j,i) = (y_ + 0.5)*face_w - 0.5;
			
			}	
		}
	}

	cv::remap(srcFrame, dstFrame, this->map_x, this->map_y, CV_INTER_CUBIC,2);

	srcFrame.release();
}


void fisheyeImgConv::equirect2persp(const cv::Mat &img, cv::Mat &dstFrame,float FOV, float THETA, float PHI, int Hd, int Wd)
{
	int equ_h{img.rows};
	int equ_w{img.cols};

	float equ_cx = equ_w/2.0;
	float equ_cy = equ_h/2.0;

	float wFOV = FOV;
	float hFOV = Hd*1.0/Wd*wFOV;

	float c_x = Wd/2.0;
	float c_y = Hd/2.0;

	// float wangle = (180 - wFOV)/2.0;
	float w_len = 2*std::tan(wFOV*CV_PI/360.0);
	float w_interva = w_len*1.0/Wd;

	// float hangle = (180 - hFOV)/2.0;
	float h_len = 2*std::tan(hFOV*CV_PI/360.0);
	float h_interva = h_len*1.0/Hd;

	this->Hd = Hd;
	this->Wd = Wd;

	this->map_x = cv::Mat::zeros(this->Hd, this->Wd, CV_32FC1);
	this->map_y = cv::Mat::zeros(this->Hd, this->Wd, CV_32FC1);

	float x_,y_,z_,r;
	cv::Mat R;
	cv::Mat XYZ = (cv::Mat_<double>(3,1) << x_,y_,z_);

	float alpha{0};
	float beta = -PHI*CV_PI/180.0;
	float gamma = THETA*CV_PI/180.0;

	// std::cout << alpha << ", " << beta << ", " << gamma;

	cv::Mat Rx = (cv::Mat_<double>(3,3) << 1,0,0,0,std::cos(alpha),-1*std::sin(alpha),0,std::sin(alpha),std::cos(alpha));
	cv::Mat Ry = (cv::Mat_<double>(3,3) << std::cos(beta),0,-1*std::sin(beta),0,1,0,std::sin(beta),0,std::cos(beta));
	cv::Mat Rz = (cv::Mat_<double>(3,3) << std::cos(gamma),-1*std::sin(gamma),0,std::sin(gamma),std::cos(gamma),0,0,0,1);
	R = Rz*Ry*Rx;

	float lat,lon;

	for(size_t x{0}; x<Wd; x++)
	{
		for(size_t y{0}; y<Hd; y++)
		{
			x_ = 1;
			y_ = (x*1.0/Wd - 0.5)*w_len;
			z_ = (y*1.0/Hd - 0.5)*h_len;

			r = std::sqrt(x_*x_ + y_*y_ + z_*z_);

			x_ = x_*1.0/r;
			y_ = y_*1.0/r;
			z_ = z_*1.0/r;

			XYZ.at<double>(0,0) = x_;
			XYZ.at<double>(1,0) = y_;
			XYZ.at<double>(2,0) = z_;

			// std::cout << "Before : " << XYZ << std::endl;
			XYZ = R*XYZ;
			// std::cout << "After : " << XYZ << std::endl;
			x_ = XYZ.at<double>(0,0);
			y_ = XYZ.at<double>(1,0);
			z_ = XYZ.at<double>(2,0);

			lon = (std::asin(z_)/CV_PI + 0.5)*equ_h;
			lat = (std::atan2(y_,x_+0.01)/(2*CV_PI) + 0.5)*equ_w;

			// lon = std::asin(z_);
			// lat = std::atan2(y_,x_);

			this->map_x.at<float>(y,x) = lat;
			this->map_y.at<float>(y,x) = lon;
		}
	}

	// std::cout << this->map_x;


	cv::remap(img, dstFrame, this->map_x, this->map_y, CV_INTER_CUBIC,2);
	// cv::imshow("output",dstFrame);
	// cv::waitKey(0);

	R.release();
	XYZ.release();

}

void fisheyeImgConv::cubemap2persp(const cv::Mat &img1, cv::Mat &dstFrame,float FOV, float THETA, float PHI, int Hd, int Wd)
{
	cv::Mat img;

	this->cubemap2equirect(img1,cv::Size (4*Hd,2*Hd), img);

	int equ_h{img.rows};
	int equ_w{img.cols};

	float equ_cx = equ_w/2.0;
	float equ_cy = equ_h/2.0;

	float wFOV = FOV;
	float hFOV = Hd*1.0/Wd*wFOV;

	float c_x = Wd/2.0;
	float c_y = Hd/2.0;

	// float wangle = (180 - wFOV)/2.0;
	float w_len = 2*std::tan(wFOV*CV_PI/360.0);
	float w_interva = w_len*1.0/Wd;

	// float hangle = (180 - hFOV)/2.0;
	float h_len = 2*std::tan(hFOV*CV_PI/360.0);
	float h_interva = h_len*1.0/Hd;

	this->Hd = Hd;
	this->Wd = Wd;

	this->map_x = cv::Mat::zeros(this->Hd, this->Wd, CV_32FC1);
	this->map_y = cv::Mat::zeros(this->Hd, this->Wd, CV_32FC1);

	float x_,y_,z_,r;
	cv::Mat R;
	cv::Mat XYZ = (cv::Mat_<double>(3,1) << x_,y_,z_);

	float alpha{0};
	float beta = -PHI*CV_PI/180.0;
	float gamma = THETA*CV_PI/180.0;

	// std::cout << alpha << ", " << beta << ", " << gamma;

	cv::Mat Rx = (cv::Mat_<double>(3,3) << 1,0,0,0,std::cos(alpha),-1*std::sin(alpha),0,std::sin(alpha),std::cos(alpha));
	cv::Mat Ry = (cv::Mat_<double>(3,3) << std::cos(beta),0,-1*std::sin(beta),0,1,0,std::sin(beta),0,std::cos(beta));
	cv::Mat Rz = (cv::Mat_<double>(3,3) << std::cos(gamma),-1*std::sin(gamma),0,std::sin(gamma),std::cos(gamma),0,0,0,1);
	R = Rz*Ry*Rx;

	float lat,lon;

	for(size_t x{0}; x<Wd; x++)
	{
		for(size_t y{0}; y<Hd; y++)
		{
			x_ = 1;
			y_ = (x*1.0/Wd - 0.5)*w_len;
			z_ = (y*1.0/Hd - 0.5)*h_len;

			r = std::sqrt(x_*x_ + y_*y_ + z_*z_);

			x_ = x_*1.0/r;
			y_ = y_*1.0/r;
			z_ = z_*1.0/r;

			XYZ.at<double>(0,0) = x_;
			XYZ.at<double>(1,0) = y_;
			XYZ.at<double>(2,0) = z_;

			// std::cout << "Before : " << XYZ << std::endl;
			XYZ = R*XYZ;
			// std::cout << "After : " << XYZ << std::endl;
			x_ = XYZ.at<double>(0,0);
			y_ = XYZ.at<double>(1,0);
			z_ = XYZ.at<double>(2,0);

			lon = (std::asin(z_)/CV_PI + 0.5)*equ_h;
			lat = (std::atan2(y_,x_+0.01)/(2*CV_PI) + 0.5)*equ_w;

			// lon = std::asin(z_);
			// lat = std::atan2(y_,x_);

			this->map_x.at<float>(y,x) = lat;
			this->map_y.at<float>(y,x) = lon;
		}
	}

	// std::cout << this->map_x;


	cv::remap(img, dstFrame, this->map_x, this->map_y, CV_INTER_CUBIC,2);
	// cv::imshow("output",dstFrame);
	// cv::waitKey(0);

	R.release();
	XYZ.release();

}


void fisheyeImgConv::equirect2Fisheye_UCM(const cv::Mat &img, cv::Mat &dstFrame, cv::Size outShape, float f, float xi,float alpha, float beta, float gamma)
{
	this->Hd = outShape.height;
	this->Wd = outShape.width;

	int Hs{img.rows};
	int Ws{img.cols};

	int Cx = this->Wd/2.0;
	int Cy = this->Hd/2.0;

	float fmin{0},omega{0},Ps_x{0},Ps_y{0},Ps_z{0},theta{0},phi{0},a{0},b{0},r{0},x_hat{0},y_hat{0},x2_y2_hat{0};

	cv::Mat R;
	cv::Mat Ps = (cv::Mat_<double>(3,1) << Ps_x,Ps_y,Ps_z);

	R = this->RMat(alpha,beta,gamma)*(this->RMat(0, -CV_PI/2, CV_PI/4)*this->RMat(0,CV_PI/2,CV_PI/2));

	// std::cout << R;

	// fmin = std::sqrt(-(1-xi*xi)*((1-Cx)*(1-Cx) + (1-Cy)*(1-Cy)))*1.0001;
	fmin = std::sqrt(std::abs(-(1-xi*xi)*((1-Cx)*(1-Cx) + (1-Cy)*(1-Cy))))*1.0001;

	// std::cout << fmin << " , " << f;

	this->map_x = cv::Mat::zeros(this->Hd, this->Wd, CV_32FC1);
	this->map_y = cv::Mat::zeros(this->Hd, this->Wd, CV_32FC1);

	for(int x{0} ; x< this->Wd; x++)
	{
		for(int y{0}; y < this->Hd; y++)
		{
			x_hat = (float(x - Cx)/f);
			y_hat = (float(y - Cy)/f);

			x2_y2_hat = x_hat*x_hat + y_hat*y_hat;


			if ((1+(1-xi*xi)*x2_y2_hat) >= 0)
			{
				omega = (xi + std::sqrt(1+(1-xi*xi)*x2_y2_hat))*1.0/(x2_y2_hat+1);
			}
			else
			{
				omega = xi*1.0/(x2_y2_hat+1);
			}

			Ps_x = omega*x_hat;
			Ps_y = omega*y_hat;
			Ps_z = omega - xi;

			Ps.at<double>(0,0) = Ps_x;
			Ps.at<double>(1,0) = Ps_y;
			Ps.at<double>(2,0) = Ps_z;

			Ps = R*Ps;

			Ps_x = Ps.at<double>(0,0);
			Ps_y = Ps.at<double>(1,0);
			Ps_z = Ps.at<double>(2,0);

			theta = std::atan2(Ps_y,Ps_x);
			phi = std::atan2(Ps_z,std::sqrt(Ps_x*Ps_x + Ps_y*Ps_y));

			a = 2*CV_PI/(Ws-1);
			b = CV_PI - a*(Ws - 1);
			this->map_x.at<float>(y,x) = (1.0/a)*(theta - b);

			a = -CV_PI/(Hs - 1);
			b = CV_PI/2;
			this->map_y.at<float>(y,x) = (1.0/a)*(phi - b);
		}
	}
	cv::remap(img, dstFrame, this->map_x, this->map_y, CV_INTER_CUBIC,2);
	
	if (f  < fmin)
	{
		r = std::sqrt(std::abs(-(f*f)/(1-xi*xi)));
		cv::Mat mask=cv::Mat::zeros(dstFrame.rows,dstFrame.cols, CV_8UC3);
		cv::circle(mask, cv::Point (Cx,Cy), r,cv::Scalar (255,255,255),-1);
		cv::bitwise_and(dstFrame, mask, dstFrame);
	}
	// cv::imshow("output",dstFrame);
	// cv::waitKey(0);
}

void fisheyeImgConv::equirect2Fisheye_EUCM(const cv::Mat &img, cv::Mat &dstFrame, cv::Size outShape, float f, float a_, float b_,float alpha, float beta, float gamma)
{
	this->Hd = outShape.height;
	this->Wd = outShape.width;

	int Hs{img.rows};
	int Ws{img.cols};

	int Cx = this->Wd/2.0;
	int Cy = this->Hd/2.0;

	float fmin{0},omega{0},Ps_x{0},Ps_y{0},Ps_z{0},theta{0},phi{0},a{0},b{0},r{0},x_hat{0},y_hat{0},x2_y2_hat{0},z_hat{0},coef{0},K_{0},del_{0};

	cv::Mat R;
	cv::Mat Ps = (cv::Mat_<double>(3,1) << Ps_x,Ps_y,Ps_z);

	R = this->RMat(alpha,beta,gamma)*(this->RMat(0, -CV_PI/2, CV_PI/4)*this->RMat(0,CV_PI/2,CV_PI/2));

	// std::cout << R;

	fmin = std::sqrt(std::abs(b_*(2*a_ - 1)*((1-Cx)*(1-Cx) + (1-Cy)*(1-Cy))))*1.0001;

	this->map_x = cv::Mat::zeros(this->Hd, this->Wd, CV_32FC1);
	this->map_y = cv::Mat::zeros(this->Hd, this->Wd, CV_32FC1);

	for(int x{0} ; x< this->Wd; x++)
	{
		for(int y{0}; y < this->Hd; y++)
		{
			x_hat = (float(x - Cx)/f);
			y_hat = (float(y - Cy)/f);
			
			x2_y2_hat = x_hat*x_hat + y_hat*y_hat;

			K_ = b_*x2_y2_hat;
			del_ = 1-(2*a_-1)*K_;

			if (del_ < 0)
				z_hat = ((1 - a_*a_*K_)*(1-a_))/(1-2*a_+a_*a_*(1+std::abs(del_)));
			else
				z_hat = (1-a_*a_*K_)/((1-a_)+a_*std::sqrt(del_));

			coef = 1/std::sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat);

			Ps_x = x_hat*coef;
			Ps_y = y_hat*coef;
			Ps_z = z_hat*coef;

			Ps.at<double>(0,0) = Ps_x;
			Ps.at<double>(1,0) = Ps_y;
			Ps.at<double>(2,0) = Ps_z;

			Ps = R*Ps;

			Ps_x = Ps.at<double>(0,0);
			Ps_y = Ps.at<double>(1,0);
			Ps_z = Ps.at<double>(2,0);

			theta = std::atan2(Ps_y,Ps_x);
			phi = std::atan2(Ps_z,std::sqrt(Ps_x*Ps_x + Ps_y*Ps_y));

			a = 2*CV_PI/(Ws-1);
			b = CV_PI - a*(Ws - 1);
			this->map_x.at<float>(y,x) = (1.0/a)*(theta - b);

			a = -CV_PI/(Hs - 1);
			b = CV_PI/2;
			this->map_y.at<float>(y,x) = (1.0/a)*(phi - b);
		}
	}
	cv::remap(img, dstFrame, this->map_x, this->map_y, CV_INTER_CUBIC,2);
	
	if (f  < fmin)
	{
		r = std::sqrt(std::abs((f*f)/(b_*(2*a_ - 1))));
		cv::Mat mask=cv::Mat::zeros(dstFrame.rows,dstFrame.cols, CV_8UC3);
		cv::circle(mask, cv::Point (Cx,Cy), r,cv::Scalar (255,255,255),-1);
		cv::bitwise_and(dstFrame, mask, dstFrame);
	}
	// cv::imshow("output",dstFrame);
	// cv::waitKey(0);
}

void fisheyeImgConv::equirect2Fisheye_FOV(const cv::Mat &img, cv::Mat &dstFrame, cv::Size outShape, float f, float w_,float alpha, float beta, float gamma)
{
	this->Hd = outShape.height;
	this->Wd = outShape.width;

	int Hs{img.rows};
	int Ws{img.cols};

	int Cx = this->Wd/2.0;
	int Cy = this->Hd/2.0;

	float fmin{0},omega{0},Ps_x{0},Ps_y{0},Ps_z{0},theta{0},phi{0},a{0},b{0},r{0},x_hat{0},y_hat{0},x2_y2_hat{0};

	cv::Mat R;
	cv::Mat Ps = (cv::Mat_<double>(3,1) << Ps_x,Ps_y,Ps_z);

	R = this->RMat(alpha,beta,gamma)*(this->RMat(0, -CV_PI/2, CV_PI/4)*this->RMat(0,CV_PI/2,CV_PI/2));

	// std::cout << R;

	// fmin = std::sqrt(-(1-xi*xi)*((1-Cx)*(1-Cx) + (1-Cy)*(1-Cy)))*1.0001;
	// fmin = std::sqrt(std::abs(-(1-xi*xi)*((1-Cx)*(1-Cx) + (1-Cy)*(1-Cy))))*1.0001;

	// std::cout << fmin << " , " << f;

	this->map_x = cv::Mat::zeros(this->Hd, this->Wd, CV_32FC1);
	this->map_y = cv::Mat::zeros(this->Hd, this->Wd, CV_32FC1);

	for(int x{0} ; x< this->Wd; x++)
	{
		for(int y{0}; y < this->Hd; y++)
		{
			x_hat = (float(x - Cx)/f);
			y_hat = (float(y - Cy)/f);

			x2_y2_hat = std::sqrt(x_hat*x_hat + y_hat*y_hat);

			Ps_x = x_hat*std::sin(x2_y2_hat*w_)/(2*x2_y2_hat*std::tan(w_/2));
			Ps_y = y_hat*std::sin(x2_y2_hat*w_)/(2*x2_y2_hat*std::tan(w_/2));
			Ps_z = std::cos(x2_y2_hat*w_);

			Ps.at<double>(0,0) = Ps_x;
			Ps.at<double>(1,0) = Ps_y;
			Ps.at<double>(2,0) = Ps_z;

			Ps = R*Ps;

			Ps_x = Ps.at<double>(0,0);
			Ps_y = Ps.at<double>(1,0);
			Ps_z = Ps.at<double>(2,0);

			theta = std::atan2(Ps_y,Ps_x);
			phi = std::atan2(Ps_z,std::sqrt(Ps_x*Ps_x + Ps_y*Ps_y));

			a = 2*CV_PI/(Ws-1);
			b = CV_PI - a*(Ws - 1);
			this->map_x.at<float>(y,x) = (1.0/a)*(theta - b);

			a = -CV_PI/(Hs - 1);
			b = CV_PI/2;
			this->map_y.at<float>(y,x) = (1.0/a)*(phi - b);
		}
	}
	cv::remap(img, dstFrame, this->map_x, this->map_y, CV_INTER_CUBIC,2);
	
	// cv::imshow("output",dstFrame);
	// cv::waitKey(0);
}

void fisheyeImgConv::equirect2Fisheye_DS(const cv::Mat &img, cv::Mat &dstFrame, cv::Size outShape, float f, float a_, float xi_,float alpha, float beta, float gamma)
{
	this->Hd = outShape.height;
	this->Wd = outShape.width;

	int Hs{img.rows};
	int Ws{img.cols};

	int Cx = this->Wd/2.0;
	int Cy = this->Hd/2.0;

	float fmin{0},omega{0},Ps_x{0},Ps_y{0},Ps_z{0},theta{0},phi{0},a{0},b{0},r{0},x_hat{0},y_hat{0},x2_y2_hat{0},z_hat{0},coef{0},K_{0},del_{0};

	cv::Mat R;
	cv::Mat Ps = (cv::Mat_<double>(3,1) << Ps_x,Ps_y,Ps_z);

	R = this->RMat(alpha,beta,gamma)*(this->RMat(0, -CV_PI/2, CV_PI/4)*this->RMat(0,CV_PI/2,CV_PI/2));

	// std::cout << R;

	fmin = std::sqrt(std::abs((2*a_ - 1)*((1-Cx)*(1-Cx) + (1-Cy)*(1-Cy))))*1.0001;

	this->map_x = cv::Mat::zeros(this->Hd, this->Wd, CV_32FC1);
	this->map_y = cv::Mat::zeros(this->Hd, this->Wd, CV_32FC1);

	for(int x{0} ; x< this->Wd; x++)
	{
		for(int y{0}; y < this->Hd; y++)
		{
			x_hat = (float(x - Cx)/f);
			y_hat = (float(y - Cy)/f);
			
			x2_y2_hat = x_hat*x_hat + y_hat*y_hat;

			del_ = 1-(2*a_-1)*x2_y2_hat;

			if (del_ < 0)
				z_hat = ((1 - a_*a_*x2_y2_hat)*(1-a_))/(1-2*a_+a_*a_*(1+std::abs(del_)));
			else
				z_hat = (1-a_*a_*x2_y2_hat)/((1-a_)+a_*std::sqrt(del_));

			del_ = z_hat*z_hat + (1 - xi_*xi_)*x2_y2_hat;

			if (del_ < 0)
				omega = x2_y2_hat*xi_/(z_hat*z_hat + x2_y2_hat);
			else
				omega = (z_hat*xi_ + std::sqrt(del_))/(z_hat*z_hat + x2_y2_hat); 

			Ps_x = omega*x_hat;
			Ps_y = omega*y_hat;
			Ps_z = omega*z_hat - xi_;

			Ps.at<double>(0,0) = Ps_x;
			Ps.at<double>(1,0) = Ps_y;
			Ps.at<double>(2,0) = Ps_z;

			Ps = R*Ps;

			Ps_x = Ps.at<double>(0,0);
			Ps_y = Ps.at<double>(1,0);
			Ps_z = Ps.at<double>(2,0);

			theta = std::atan2(Ps_y,Ps_x);
			phi = std::atan2(Ps_z,std::sqrt(Ps_x*Ps_x + Ps_y*Ps_y));

			a = 2*CV_PI/(Ws-1);
			b = CV_PI - a*(Ws - 1);
			this->map_x.at<float>(y,x) = (1.0/a)*(theta - b);

			a = -CV_PI/(Hs - 1);
			b = CV_PI/2;
			this->map_y.at<float>(y,x) = (1.0/a)*(phi - b);
		}
	}
	cv::remap(img, dstFrame, this->map_x, this->map_y, CV_INTER_CUBIC,2);
	// std::cout << "f = " << f << "| fmin = " << fmin;
	if (f  < fmin)
	{
		r = std::sqrt(std::abs((f*f)/(2*a_ - 1)));
		cv::Mat mask=cv::Mat::zeros(dstFrame.rows,dstFrame.cols, CV_8UC3);
		cv::circle(mask, cv::Point (Cx,Cy), r,cv::Scalar (255,255,255),-1);
		cv::bitwise_and(dstFrame, mask, dstFrame);
	}
	// cv::imshow("output",dstFrame);
	// cv::waitKey(0);
}
