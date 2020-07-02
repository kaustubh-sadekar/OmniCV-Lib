#include<iostream>
#include<opencv2/opencv.hpp>
#include"../omnicv/utils.hpp"
#include <opencv2/core/core.hpp>

int main()
{
	fisheyeImgConv mapper1;

	std::cout << "\n\n########################################################\n";
	std::cout << "Running visual tests for C++ library ...." << std::endl;

	cv::namedWindow("output image",cv::WINDOW_NORMAL);
	cv::resizeWindow("output image",400,400);
	cv::namedWindow("reference image",cv::WINDOW_NORMAL);
	cv::resizeWindow("reference image",400,400);

	std::string img_path = "./input/equirect.jpg";

	cv::Mat equiRect,fisheye;

	equiRect = cv::imread(img_path);
	assert(equiRect.cols != 0 && "Wrong path... Run the tests.py code after changing the directory inside test folder");

	std::cout << "[1] Testing equirect2Fisheye_UCM method ..."<<std::endl;
	mapper1.equirect2Fisheye_UCM(equiRect,fisheye, cv::Size (250,250),50,0.5);
	img_path = "./outputs/UCM_out.jpg";
	cv::Mat refImg = cv::imread(img_path);
	cv::imshow("output image",fisheye);
	cv::imshow("reference image",refImg);
	cv::waitKey(0);
	assert (refImg.cols !=0 && "Output reference image not present");
	assert ((fisheye.cols==250) && (fisheye.rows==250) &&"Output dimensions for generated fisheye image do not match !");

	std::cout << "[2] Testing equirect2Fisheye_EUCM method ..." << std::endl;
	mapper1.equirect2Fisheye_EUCM(equiRect,fisheye, cv::Size (250,250),100,0.4,2);
	img_path = "./outputs/EUCM_out.jpg";
	refImg = cv::imread(img_path);
	cv::imshow("output image",fisheye);
	cv::imshow("reference image",refImg);
	cv::waitKey(0);
	assert (refImg.cols !=0 && "Output reference image not present");
	assert ((fisheye.cols==250) && (fisheye.rows==250) &&"Output dimensions for generated fisheye image do not match !");

	std::cout << "[3] Testing equirect2Fisheye_FOV method ..." << std::endl;
	mapper1.equirect2Fisheye_FOV(equiRect,fisheye, cv::Size (250,250),40,0.5);
	img_path = "./outputs/FOV_out.jpg";
	refImg = cv::imread(img_path);
	cv::imshow("output image",fisheye);
	cv::imshow("reference image",refImg);
	cv::waitKey(0);
	assert (refImg.cols !=0 && "Output reference image not present");
	assert ((fisheye.cols==250) && (fisheye.rows==250) &&"Output dimensions for generated fisheye image do not match !");

	std::cout << "[4] Testing equirect2Fisheye_DS method ..." << std::endl;
	mapper1.equirect2Fisheye_DS(equiRect,fisheye, cv::Size (250,250),90,0.4,0.8);
	img_path = "./outputs/DS_out.jpg";
	refImg = cv::imread(img_path);
	cv::imshow("output image",fisheye);
	cv::imshow("reference image",refImg);
	cv::waitKey(0);
	assert (refImg.cols !=0 && "Output reference image not present");
	assert ((fisheye.cols==250) && (fisheye.rows==250) &&"Output dimensions for generated fisheye image do not match !");

	std::string params_file_path = "../fisheyeParams.txt";
	
	mapper1.filePath = "./input/fisheyeParams.txt";

	img_path = "./input/fisheye.jpg";
	fisheye = cv::imread(img_path);
	assert(fisheye.cols != 0 && "Wrong path... Run the tests.py code after changing the directory inside test folder");

	std::cout << "[5] Testing fisheye2equirect method for mode=1..." << std::endl;
	mapper1.fisheye2equirect(fisheye,equiRect, cv::Size (400,200));
	img_path = "./outputs/f2e_out.jpg";
	refImg = cv::imread(img_path);
	cv::imshow("output image",equiRect);
	cv::imshow("reference image",refImg);
	cv::waitKey(0);
	assert (refImg.cols !=0 && "Output reference image not present");
	assert ((equiRect.cols==400) && (equiRect.rows==200) &&"Output dimensions for generated equirectangular image do not match !");

	//Mode 2
	cv::Mat cubemap;
	std::cout << "[6] Testing fisheye2equirect method for mode=2..." << std::endl;
	mapper1.fisheye2equirect(fisheye,equiRect, cv::Size (400,200));
	mapper1.eqrect2cubemap(equiRect,cubemap,256,true,true);
	mapper1.cubemap2equirect(cubemap,cv::Size (400,200),equiRect);
	img_path = "./outputs/f2e_out_mode2.jpg";
	refImg = cv::imread(img_path);
	cv::imshow("output image",equiRect);
	cv::imshow("reference image",refImg);
	cv::waitKey(0);
	assert (refImg.cols !=0 && "Output reference image not present");
	assert ((equiRect.cols==400) && (equiRect.rows==200) &&"Output dimensions for generated equirectangular image do not match !");

	img_path = "./input/equirect.jpg";
	equiRect = cv::imread(img_path);
	assert(equiRect.cols != 0 && "Wrong path... Run the tests.py code after changing the directory inside test folder");

	std::cout << "[7] Testing equirect2cubemap method for horizontal mode..." << std::endl;
	mapper1.eqrect2cubemap(equiRect,cubemap,256);
	img_path = "./outputs/e2c_out.jpg";
	refImg = cv::imread(img_path);
	cv::imshow("output image",cubemap);
	cv::imshow("reference image",refImg);
	cv::waitKey(0);
	assert (refImg.cols !=0 && "Output reference image not present");
	assert ((cubemap.cols==1536) && (cubemap.rows==256) &&"Output dimensions for generated cubemap image do not match !");

	std::cout << "[8] Testing equirect2cubemap method for dice mode..." << std::endl;
	mapper1.eqrect2cubemap(equiRect,cubemap,256,false,true);
	img_path = "./outputs/e2c_out_dice.jpg";
	refImg = cv::imread(img_path);
	cv::imshow("output image",cubemap);
	cv::imshow("reference image",refImg);
	cv::waitKey(0);
	assert (refImg.cols !=0 && "Output reference image not present");
	assert ((cubemap.cols==1024) && (cubemap.rows==768) &&"Output dimensions for generated cubemap image do not match !");

	cv::Mat persp;
	std::cout << "[9] Testing eqruirect2persp method ..." << std::endl;
	mapper1.equirect2persp(equiRect,persp,90,0,0,400,400);
	img_path = "./outputs/e2p_out.jpg";
	refImg = cv::imread(img_path);
	cv::imshow("output image",persp);
	cv::imshow("reference image",refImg);
	cv::waitKey(0);
	assert (refImg.cols !=0 && "Output reference image not present");
	assert ((persp.cols==400) && (persp.rows==400) &&"Output dimensions for generated perspective image do not match !");

	img_path = "./input/cubemap_dice.jpg";
	cubemap = cv::imread(img_path);
	assert(cubemap.cols != 0 && "Wrong path... Run the tests.py code after changing the directory inside test folder");

	std::cout << "[10] Testing cubemap2equirect method for horizontal mode..." << std::endl;
	mapper1.cubemap2equirect(cubemap,cv::Size (400,200), equiRect);
	img_path = "./outputs/c2e_out_dice.jpg";
	refImg = cv::imread(img_path);
	cv::imshow("output image",equiRect);
	cv::imshow("reference image",refImg);
	cv::waitKey(0);
	assert (refImg.cols !=0 && "Output reference image not present");
	assert ((equiRect.cols==400) && (equiRect.rows==200) &&"Output dimensions for generated equirectangular image do not match !");

	img_path = "./input/cubemap_dice.jpg";
	cubemap = cv::imread(img_path);
	assert(cubemap.cols != 0 && "Wrong path... Run the tests.py code after changing the directory inside test folder");

	std::cout << "[11] Testing cubemap2persp method for dice mode..." << std::endl;
	mapper1.cubemap2persp(cubemap,persp,90,0,0,400,400);
	img_path = "./outputs/c2p_out_dice.jpg";
	refImg = cv::imread(img_path);
	cv::imshow("output image",persp);
	cv::imshow("reference image",refImg);
	cv::waitKey(0);
	assert (refImg.cols !=0 && "Output reference image not present");
	assert ((persp.cols==400) && (persp.rows==400) &&"Output dimensions for generated perspective image do not match !");

	std::cout << "All tests completed ...";

	return 0;
}