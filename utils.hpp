#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

class fisheyeImgConv
{

    public:
    	
        std::string filePath;
        int Wd; 
        int Hd;
        int Ws;
        int Hs;
        cv::Mat map_x;
        cv::Mat map_y;
        bool singleLens;
        int Cx;
        int Cy;
        int radius;
        int aperture;
        bool dice;
        int side;
        


        fisheyeImgConv(std::string paramFilePath="None");
        void fisheye2equirect(const cv::Mat &srcFrame, cv::Mat &dstFrame, cv::Size outShape, int aperture=0, int delx=0, int dely=0, int radius=0, bool edit_mode=false);
        void meshgrid(const cv::Range &xgv, const cv::Range &ygv,cv::Mat &X, cv::Mat &Y);
        void applyMap(const int &map, const cv::Mat &srcFrame, cv::Mat &dstFrame);
        void eqrect2cubemap(const cv::Mat &srcFrame,cv::Mat &dstFrame, const int side=256, const bool modif=false, const bool dice = false);
        void cubemap2equirect(const cv::Mat &srcFrame1, const cv::Size outShape, cv::Mat &outFrame);
        void equirect2persp(const cv::Mat &img, cv::Mat &dstFrame,float FOV, float THETA, float PHI, int Hd, int Wd);
        void cubemap2persp(const cv::Mat &img, cv::Mat &dstFrame, float FOV, float THETA, float PHI, int Hd, int Wd);
        void equirect2Fisheye_UCM(const cv::Mat &img, cv::Mat &dstFrame, cv::Size outShape, float f=50, float xi=1.2,float alpha=0, float beta=0, float gamma=0);
        void equirect2Fisheye_EUCM(const cv::Mat &img, cv::Mat &dstFrame, cv::Size outShape, float f=50, float a_ = 0.5, float b_=0.5, float alpha=0, float beta=0, float gamma=0);
        void equirect2Fisheye_FOV(const cv::Mat &img, cv::Mat &dstFrame, cv::Size outShape, float f=50, float w_=1.2,float alpha=0, float beta=0, float gamma=0);
        void equirect2Fisheye_DS(const cv::Mat &img, cv::Mat &dstFrame, cv::Size outShape, float f=50, float a_ = 0.5, float xi_=0.5, float alpha=0, float beta=0, float gamma=0);
        cv::Mat RMat(double alpha, double beta, double gamma);
};
