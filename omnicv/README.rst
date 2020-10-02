OmniCV Library
==============

A **computer vision library for omnidirectional(360-degree) cameras**. This package is divided into two parts:
* **Basic functions** for inter-conversion of different types of mappings associated with omni directional cameras, virtual reality and 360-degree; videos, like cubemap, spherical projections, perspective projection and equirectangular projection.
* **Software applications** like **360-degree; video viewer**, **fisheye image generator** with variable intrinsic properties, GUI to determine **fisheye camera paraeters**.


Objectives of the OmniCV library
================================

This library has been developed with the following obectives:

* **Quick and easy** to use API to encourage and enhance the research in areas using omni directional cameras.
* To support **real time** applications.
* Provide extensions in **python** as well as **C++** as they are the languages used by researchers.
* Provide **ROS package** to use in robotics research.

Click `here <https://kaustubh-sadekar.github.io/OmniCV-Lib/index.html>`_ to view the documentation page. There are several examples and other details shared on the documentation page.

Installation guide
==================

A custom make file has been written which provides quick and easy options for installing and testing the library.

.. code:: shell

    git clone https://github.com/kaustubh-sadekar/OmniCV-Lib
    cd OmniCV-Lib/omnicv/
    # To build c++ as well as python files
    make build-all
    # To build only python files
    make build-python
    # To build only c++ files
    make build-cpp

Installing OmniCV in a virtual environment using pipenv. Pipfile and Pipfile.lock files have been provided. Copy both the files to the present working directory. Then simply run the following commands to setup OmniCV in a local environment.

.. code:: shell

    pipenv install
    pipenv shell    


Running Tests
=============

There are two types of tests provided for users. Being a vision based package visual tests have also been provided.

To run non-visual tests

.. code:: shell

    cd OmniCV-Lib/omnicv/
    # To test only python extension of the project
    make test-py
    # To test only c++ extension of the project
    make test-cpp
    # To test python as well as c++ extension of the project
    make test-cpp


Examples are available `here <https://kaustubh-sadekar.github.io/OmniCV-Lib/index.html>`_ on the official documentation page.

