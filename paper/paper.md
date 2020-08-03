---
title: 'OmniCV - A library for Omnidirectional Cameras'
tags:
  - Omnidirectional cameras
  - Euirectangular images
  - Cubemap
  - Fisheye camera
  - 360 degree videos
authors:
 - name: Kaustubh Sadekar
   affiliation: "1"
 - name: Leena Vachhani
   affiliation: "1"
 - name: Abhishek Gupta
   affiliation: "2"
affiliations:
 - name: IDP in Systems and Control Engineering, IIT Bombay, India
   index: 1
 - name: Department of Mechanical Engineering, IIT Bombay, India
   index: 2
date: 10 June 2020
bibliography: paper.bib
---

# Summary 
Omnidirectional cameras refer to cameras with large field-of-view (FOV), more than 180°, and ideally 360°. This includes cameras with horizontal 360° field-of-view(FOV) and cameras with FOV spanning a 360° horizontal and more than 90 (ideally 180) vertical field-of-view(FOV). It is beneficial to use such large field-of-view cameras, especially for indoor environments (@Scaramuzza2014). There has been a constant growth in research related to omnidirectional cameras (@Zhang2016BenefitOL). Most of the applications using such cameras either need a mathematical model of the camera or some form of mapping to represent the 360° image information like equirectangular projection or cubemap projection.

OmniCV is a complete library that is written to support and enhance the research related to omnidirectional cameras. It contains, different class methods and APIs to, simulate virtual cameras based on different omnidirectional camera models and also contains methods for interconversion of different representations of a 360° image like equirectangular projection, cubemap projection and perspective projection. The library also contains some essential GUI based software written using the package APIs that enable the user to determine various parameters of the camera used to capture any given image. All the functionalities are written in C++ as well as Python. The operations are optimized to ensure significant use in real-time applications. Moreover, ROS nodes for all the functionalities are also provided for easy integration with any existing ROS project. 

Several applications and research projects related to omnidirectional cameras have used a few of the above-mentioned functionalities and have also open-sourced their code but there is no single complete open-source library available with C++, Python and ROS compatibility. In “DeepCalib: A deep learning approach for automatic intrinsic calibration of wide field-of-view cameras” (@inproceedings), a virtual camera based on the unified spherical model (@Barreto2006) is used to generate multiple synthetic images, with different intrinsic parameters, for each 360° image of the sun360 dataset. In “Flat2Sphere: Learning Spherical Convolution for Fast Features from 360° Imagery” (@NIPS2017_6656) inverse perspective projection to equirectangular projections at different polar angles is used to test the model on PASCAL VOC dataset. In ” CubemapSLAM: A Piecewise-Pinhole Monocular Fisheye SLAM System” (@wang2018cubemapslam) the SLAM problem for a fisheye camera is addressed by converting the fisheye camera image to cubemap representation. There are some open-source projects like (@fisheyedwarp) and (@py360convert) which provide Python-based implementations for a few interconversion of 360° images from one format to the other but there is no such complete library which provides different camera models, ROS nodes and optimized implementations for various forms of interconversions of 360° images with C++ as well as Python support. Several omnidirectional camera models like Unified Camera Model (@UCM), Extended Unified Camera Model (@EUCM), Field-Of-View camera model(@FOV) as well as Double Sphere camera model (@usenko18double-sphere) have been implemented in OmniCV to support research related to omnidirectional cameras using multiple types of camera models rather than depending on just a single model.

OmniCV library has been developed with the following objectives: 1) Quick and easy to use API to encourage and enhance the research in areas using omnidirectional cameras. 2) To support real-time applications. 3) Provide extensions in Python as well as C++ as they are the languages used by researchers. 4) Provide a ROS package to use in robotics research. 
Detailed documentation is also provided, with stepwise instructions to install the library and detailed examples for each function are provided to make it easy to understand and use. A modular object-oriented programming method is used for the library, making it easily compatible and manageable for future updates. 

# Acknowledgments 
This work has made use of the following softwares : 
numpy (@NumPy) , matplotlib (@Matplotlib), jupyter (@soton403913) and OpenCV (@opencv).
Images from the fisheye dataset(@FisheyeDataset) were used in the experiments related to fisheye images.

This work is funded by a project under the National Center of Excellence in Technology and Internal Security (NCETIS), IIT Bombay sponsored by the Ministry of Electronics and Information Technology.

# References

