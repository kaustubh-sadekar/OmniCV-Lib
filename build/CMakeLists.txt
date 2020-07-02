# Defining the minimum CMake version required which can execute this file.
cmake_minimum_required(VERSION 2.8.12)

# Setting variable corresponding to C++ version to be used while compiling
set(CMAKE_CXX_STANDARD 14)

# Finding the header and library folder for OpenCV
find_package( OpenCV REQUIRED )

# Sources files
set(SOURCE_FILES
  ../omnicv/imgMap.cpp
)

set(HEADER_FILES
  ../omnicv/utils.hpp
)


# Include the headers folder for OpenCV
include_directories( ${OpenCV_INCLUDE_DIRS})

ADD_EXECUTABLE(main_output ../main_code.cpp ${SOURCE_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(main_output ${OpenCV_LIBS})

ADD_EXECUTABLE(fisheye2equirect ../Examples/fisheye2equirect.cpp ${SOURCE_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(fisheye2equirect ${OpenCV_LIBS})

ADD_EXECUTABLE(equirect2cubemap ../Examples/equirect2cubemap.cpp ${SOURCE_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(equirect2cubemap ${OpenCV_LIBS})

ADD_EXECUTABLE(cubemap2equirect ../Examples/cubemap2equirect.cpp ${SOURCE_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(cubemap2equirect ${OpenCV_LIBS})

ADD_EXECUTABLE(equirect2persp ../Examples/equirect2persp.cpp ${SOURCE_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(equirect2persp ${OpenCV_LIBS})

ADD_EXECUTABLE(cubemap2persp ../Examples/cubemap2persp.cpp ${SOURCE_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(cubemap2persp ${OpenCV_LIBS})

ADD_EXECUTABLE(equirect2fisheye ../Examples/equirect2fisheye.cpp ${SOURCE_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(equirect2fisheye ${OpenCV_LIBS})

ADD_EXECUTABLE(../tests/test ../tests/test.cpp ${SOURCE_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(../tests/test ${OpenCV_LIBS})

ADD_EXECUTABLE(../tests/test_visual ../tests/test_visual.cpp ${SOURCE_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(../tests/test_visual ${OpenCV_LIBS})

