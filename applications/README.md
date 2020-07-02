# Some application notes or use cases of the library

These applications notes help the users to understand how easy it is to create some interesting applications using this library with just a fe lines of code.

## Click on respective link to know more about the application

## [1. GUI to control focus, distortion and view orientation to generate different kinds of distortion effects and get images with different properties.](fisheyeGUI.md)

This application is a small GUI that helps you to generate images with variable focal length and distortion coefficient from a given equirectangular image.
It can be used to create some interesting visual effects which are shown below in the gif.

<p align="center">
  <img src="/gifs/eqrect2Fisheye2.gif">
</p>


## [2. GUI to view an equirectangular image in 360&deg; format with control trackbars for FOV and viewing angles. (You can download any 360&deg; video from youtube and view it using the GUI and enjoy the 360&deg; viewing experience).](viewertoyGUI.md)

This application is a small GUI which can be used to change the FOV and viewing angles to view any 360&deg; video. NOTE : This is just a toy example and not a full fledged software. It does not process any audio. Main objective of creating this GUI was to show different outputs one can generate from a equirectangular image based on various parameters.

<p align="center">
  <img src="/gifs/360Viewer.gif">
</p>


## [3. GUI to determine fisheye camera parameters like aperture and fisheye radius for further conversions.](paramsGUI.md)

This is an important use case which any user needs to use once to get the parameters of the fisheye camera like aperture and aperent radius in the image. These parameters are used in conversion of fisheye image to any other type of image. The GUI makes it easy for users to determine these parameters for fisheye image/video from an unknown camera.

GUI to get the aperent radius             |  GUI to get fisheye parameters 
:-------------------------:|:-------------------------:
![](/gifs/getRadius.gif)  |  ![](/gifs/FisheyeParams.gif)

