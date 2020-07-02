# Example for users to understand how to use the library

Different methods have been written for inter conversion between different mappings related to omni directional cameras.
## [Click here](https://kaustubh-sadekar.github.io/OmniCV-Lib/Examples.html) for a detailed API documentation and examples.

## 1. Equirectangular image to fisheye or pinhole camera image
#### Possibile areas of application
* Generate images with different focal length and distortion coefficients.
* Generate synthetic images for training of various deep learning models.
* Image augmentation
* Creating interesting visual effects with 360&deg; videos.
Some of the above mentioned applications have been discussed in detail in the application notes sections.

<p align="center">
  <img height="200" src="/gifs/eqrect2fisheye.gif">
</p>
<p align="center">
  <i>Example of conversion of equirectangular image to fisheye image</i>
</p>


## 2. Fisheye image to equirectangular image
#### Possible areas of applications
* Used for stitching and creating a 360&deg; video from two fisheye cameras facing away from eachother.
* To upload omnidirectional camera videos to youtube which can support the 360&deg; viewer GUI.
* Used as an interconversion step

<p align="center">
  <img height="200" src="/gifs/fisheye2eqrect1.gif">
</p>
<p align="center">
  <i>Example of conversion of fisheye image to equirectangular image</i>
</p>


## 3. Equirectangular image to cube map image (horizontal or dice format)
#### Possible areas of applications
* Useful to create cubemap environment textures, for video games or VR application, from a given equirectangular image.
* To remove distortions generated due to the fisheye lens.
* Used in visual SLAM methods where omni directional cameras are used.

<p align="center">
  <img height="200" src="/gifs/equirect2cubemap_dice.gif">
</p>
<p align="center">
  <i>Example of conversion of equirectangular image to cubemap image</i>
</p>


## 4. Cubemap image to equirectangular image
#### Possible areas of applications
* Create compressed equirectangular form of texture maps which can be later converted to cubemap and used as texture maps.
* Intermediate step when using omnidirectional cameras.
* Convert cubemap based images into a format that can be uploaded to a 360&deg; image/video viewer.
<p align="center">
  <img height="200" src="/gifs/cube2eqrect_dice.gif">
</p>
<p align="center">
  <i>Example of conversion of cubemap to equirectangular image</i>
</p>


## 5. Convert equirectangular image to perspective image with desired field of view and viewing angles
#### Possible areas of applications 
* Creating different perspective views in a 360&deg; video/image.
* Object detection in 360&deg; videos with existing pre-trained models.
* Creating a GUI to view 360&deg; videos and images.
* Surveillance robots using omni directional cameras.

<p align="center">
  <img height="200" src="/gifs/eqrect2persp.gif">
</p>
<p align="center">
  <i>Example of conversion of equirectangular image to perspective image</i>
</p>



## 6. Convert cubemap image to perspective image with desired field of view and viewing angles
#### Possible areas of applications 
* Create different perspective views in a 360&deg; video/image.
* Creating a GUI to view 360&deg; videos and images.
* Simulate and analyse a texture map in a video game.

<p align="center">
  <img height="200" src="/gifs/cubemap2persp_dice.gif">
</p>
<p align="center">
  <i>Example of conversion of cubemap image to perspective image</i>
</p>
