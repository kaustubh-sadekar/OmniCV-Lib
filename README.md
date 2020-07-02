<p align="center">
  <img width="250" src="logo.png">
</p>

A **computer vision library for omnidirectional(360&deg;) cameras**. This package is divided into two parts:
* **Basic functions** for inter-conversion of different types of mappings associated with omni directional cameras, virtual reality and 360&deg; videos, like cubemap, spherical projections, perspective projection and equirectangular projection.
* **Software applications** like **360&deg; video viewer**, **fisheye image generator** with variable intrinsic properties, GUI to determine **fisheye camera paraeters**.

##### Objectives of the OmniCV library
This library has been developed with the following obectives:
* **Quick and easy** to use API to encourage and enhance the research in areas using omni directional cameras.
* To support **real time** applications.
* Provide extensions in **python** as well as **C++** as they are the languages used by researchers.
* Provide **ROS package** to use in robotics research.

[*Click here to know more about omni directional cameras*](omnidir-cameras.md)

## Highlights 
* Detailed [examples](Examples/README.md) and [application notes](applications/README.md) to understand how to use the package.
* Installation guide for 
  * Python version
  * C++ version
  * ROS packge
* [**Detailed documentation**](https://kaustubh-sadekar.github.io/OmniCV-Lib/).

## Output Gallery

Some interesting 360&deg; video effects

Arround the world effect             |  Hollow world effect 
:-------------------------:|:-------------------------:
![](gifs/eqrect2FisheyeFet2.gif)  |  ![](gifs/eqrect2FisheyeFet1.gif)

Creating custom fisheye images 

Equirect2Fisheye             |  Custom image using GUI 
:-------------------------:|:-------------------------:
![](gifs/eqrect2fisheye.gif)  |  ![](gifs/eqrect2Fisheye.gif)

GUI to determine fisheye camera parameters

GUI to get radius        |  GUI to get fisheye params
:-------------------------:|:-------------------------:
![](gifs/getRadius.gif)  |  ![](gifs/FisheyeParams.gif)

Horizontal and vertical orientation viewing mode support

360&deg; viewer mode 1        |  360&deg; viewer mode 2
:-------------------------:|:-------------------------:
![](gifs/360Viewer2.gif)  |  ![](gifs/360Viewer3.gif)


## Index to theory, examples and application notes
* [Types of omni directional cameras](omnidir-cameras.md)
* [Examples](Examples/README.md)
  * Equirectangular image to fisheye or pinhole camera image.
  * Fisheye image to equirectangular image.
  * Equirectangular image to cube map image (horizontal or dice format)
  * Convert cube map image (horizontal or dice format) to equirectangular image.
  * Convert equirectangular image to perspective image with desired field of view and viewing angles.
  * Convert cubemap image to perspective image with desired field of view and viewing angles.
* [Application notes](applications/README.md)
  * GUI to control focus, distortion and view orientation to generate different kinds of distortion effects and get images with different properties.
  * GUI to determine fisheye camera parameters like aperture and fisheye radius for further conversions.
  * GUI to view an equirectangular image in 360&deg; format with control trackbars to change FOV(Field Of View) and viewing angles. (You can download any 360&deg; video **from youtube** and view it using the GUI and **enjoy the 360&deg; viewing experience**).

## Installation guide
##### Python version
The python version of the package is available on PyPi and can be installed using pip.
```shell
pip3 install requirements.txt
pip3 install omnicv
```

To download it from source 
* Clone the repository
* Run the setup.py file
```shell
git clone https://github.com/kaustubh-sadekar/OmniCV.git
cd OmniCV/
sudo python3 setup.py install
```
**Run any [python example](Examples/README.md) to check the code**

##### C++ version
To install the C++ extensions follow the below steps.
* Clone the repository
* cd to build directory
* build the files using cmake
* Include `imgMap.cpp, utils.hpp` in the `CMakeLists.txt`.
```shell
git clone https://github.com/kaustubh-sadekar/OmniCV.git
cd OmniCV/build
cmake CMakeLists.txt
make
cd ..
# run the built code (here main_code.cpp)
./build/main_code
```
##### ROS Nodes
To build the ROS nodes follow these steps:
* Create a folder named omnicv in your ros workspace where you have your other ros packages

```shell
roscd src/
mkdir omnicv
```
* Add the contents present inside ros_files folder to the omnicv folder created in the previous step.
* Build the workspace.

```shell
cp OmniCV/ros_files/ [PATH TO ROS WORKSPACE]/src/omnicv/
roscd
catkin_make
```

#### NOTE : Images labeled under `labeled to reuse` have been used for the experiments.
