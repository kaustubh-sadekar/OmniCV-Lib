#include<iostream>
#include <fstream>
#include<opencv2/opencv.hpp>
#include"../omnicv/utils.hpp"
#include <opencv2/core/core.hpp>
#define CATCH_CONFIG_MAIN
#include "../omnicv/catch.hpp"
#define REQUIRE_MESSAGE(cond, msg) do { INFO(msg); REQUIRE(cond); } while((void)0, 0)

TEST_CASE("Test")
{
	fisheyeImgConv mapper1;
        std::string params_file_path = "./input/fisheyeParams.txt";
        std::cout << ".";
	ifstream ifile;
	ifile.open(params_file_path);
	REQUIRE_MESSAGE(ifile,"Parameter file does not exist in the input folder");
	
	ifile.close();
	mapper1.filePath = params_file_path;
	std::string img_path = "./input/equirect.jpg";

	cv::Mat equiRect,fisheye;
	equiRect = cv::imread(img_path);
        std::cout << ".";
        REQUIRE_MESSAGE((equiRect.cols != 0),"Wrong path... Run the test after changing the directory inside test folder");
	
        std::cout << ".";
	mapper1.equirect2Fisheye_UCM(equiRect,fisheye, cv::Size (250,250),50,0.5);
        REQUIRE_MESSAGE(((fisheye.cols==250) && (fisheye.rows==250)),"Output dimensions for generated fisheye image do not match for equirect2Fisheye_UCM method !");
	
	std::cout << ".";
	mapper1.equirect2Fisheye_EUCM(equiRect,fisheye, cv::Size (250,250),100,0.4,2);
        REQUIRE_MESSAGE(((fisheye.cols==250) && (fisheye.rows==250)),"Output dimensions for generated fisheye image do not match equirect2Fisheye_EUCM method !");

	std::cout << ".";
	mapper1.equirect2Fisheye_FOV(equiRect,fisheye, cv::Size (250,250),40,0.5);
	REQUIRE_MESSAGE(((fisheye.cols==250) && (fisheye.rows==250)),"Output dimensions for generated fisheye image do not match equirect2Fisheye_FOV method !");
        
	std::cout << ".";
	mapper1.equirect2Fisheye_DS(equiRect,fisheye, cv::Size (250,250),90,0.4,0.8);
	REQUIRE_MESSAGE(((fisheye.cols==250) && (fisheye.rows==250)),"Output dimensions for generated fisheye image do not match equirect2Fisheye_DS method !");
        
	img_path = "./input/fisheye.jpg";
	fisheye = cv::imread(img_path);
	std::cout << ".";
        REQUIRE_MESSAGE(fisheye.cols != 0,"Wrong path... Run the tests.py code after changing the directory inside test folder");
        
	std::cout << ".";
	mapper1.fisheye2equirect(fisheye,equiRect, cv::Size (400,200));
	REQUIRE_MESSAGE(((equiRect.cols==400) && (equiRect.rows==200)),"Output dimensions for generated fisheye image do not match for fisheye2equirect method for mode=1");
	
	cv::Mat cubemap;
	std::cout << ".";
	mapper1.fisheye2equirect(fisheye,equiRect, cv::Size (400,200));
	mapper1.equirect2cubemap(equiRect,cubemap,256,true,true);
	mapper1.cubemap2equirect(cubemap,cv::Size (400,200),equiRect);
	REQUIRE_MESSAGE(((equiRect.cols==400) && (equiRect.rows==200)),"Output dimensions for generated equirectangular image do not match !");

	img_path = "./input/equirect.jpg";
	equiRect = cv::imread(img_path);
	std::cout << ".";
	REQUIRE_MESSAGE(equiRect.cols != 0,"Wrong path... Run the tests.py code after changing the directory inside test folder");

	std::cout << ".";
	mapper1.equirect2cubemap(equiRect,cubemap,256);
	REQUIRE_MESSAGE(((cubemap.cols==1536) && (cubemap.rows==256)), "Output dimensions for generated cubemap image do not match !");

	std::cout << ".";
	mapper1.equirect2cubemap(equiRect,cubemap,256,false,true);
	REQUIRE_MESSAGE(((cubemap.cols==1024) && (cubemap.rows==768)), "Output dimensions for generated cubemap image do not match !");

	cv::Mat persp;
	std::cout << ".";
	mapper1.equirect2persp(equiRect,persp,90,0,0,400,400);
	REQUIRE_MESSAGE(((persp.cols==400) && (persp.rows==400)), "Output dimensions for generated perspective image do not match !");

	img_path = "./input/cubemap.jpg";
	cubemap = cv::imread(img_path);
	std::cout << ".";
	REQUIRE_MESSAGE(cubemap.cols != 0,"Wrong path... Run the tests.py code after changing the directory inside test folder");

	std::cout << ".";
	mapper1.cubemap2equirect(cubemap,cv::Size (400,200), equiRect);
	REQUIRE_MESSAGE(((equiRect.cols==400) && (equiRect.rows==200)), "Output dimensions for generated equirectangular image do not match !");

	img_path = "./input/cubemap_dice.jpg";
	cubemap = cv::imread(img_path);
	std::cout << ".";
	REQUIRE_MESSAGE(cubemap.cols != 0, "Wrong path... Run the tests.py code after changing the directory inside test folder");

	std::cout << ".";
	mapper1.cubemap2equirect(cubemap,cv::Size (400,200), equiRect);
	REQUIRE_MESSAGE(((equiRect.cols==400) && (equiRect.rows==200)), "Output dimensions for generated equirectangular image do not match !");

	img_path = "./input/cubemap.jpg";
	cubemap = cv::imread(img_path);
	std::cout << ".";
	REQUIRE_MESSAGE(cubemap.cols != 0, "Wrong path... Run the tests.py code after changing the directory inside test folder");

	std::cout << ".";
	mapper1.cubemap2persp(cubemap,persp,90,0,0,400,400);
	REQUIRE_MESSAGE(((persp.cols==400) && (persp.rows==400)), "Output dimensions for generated perspective image do not match !");

	img_path = "./input/cubemap_dice.jpg";
	cubemap = cv::imread(img_path);
	std::cout << ".";
	REQUIRE_MESSAGE(cubemap.cols != 0, "Wrong path... Run the tests.py code after changing the directory inside test folder");

	std::cout << "." <<std::endl;
	mapper1.cubemap2persp(cubemap,persp,90,0,0,400,400);
	REQUIRE_MESSAGE(((persp.cols==400) && (persp.rows==400)),"Output dimensions for generated perspective image do not match !");

}
