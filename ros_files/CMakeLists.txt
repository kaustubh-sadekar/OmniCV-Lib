cmake_minimum_required(VERSION 2.8.3)

set(CMAKE_CXX_STANDARD 14)

project(omnicv)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
  image_transport
)

find_package(OpenCV REQUIRED)

catkin_package(
  INCLUDE_DIRS
  CATKIN_DEPENDS roscpp
                 sensor_msgs
                 cv_bridge
                 image_transport
)

include_directories(
  include
  ${OpenCV_INCLUDE_DIR}
  ${catkin_INCLUDE_DIR}
)

# Sources files
set(SOURCE_FILES
  src/imgMap.cpp
)

set(HEADER_FILES
  src/utils.hpp
)


ADD_EXECUTABLE(main_output src/main_code.cpp ${SOURCE_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(main_output ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

ADD_EXECUTABLE(vision_node src/vision_node.cpp ${SOURCE_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(vision_node ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

ADD_EXECUTABLE(equirect2fisheye src/equirect2fisheye.cpp ${SOURCE_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(equirect2fisheye ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

ADD_EXECUTABLE(fisheye2equirect src/fisheye2equirect.cpp ${SOURCE_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(fisheye2equirect ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

ADD_EXECUTABLE(equirect2cubemap src/equirect2cubemap.cpp ${SOURCE_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(equirect2cubemap ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

ADD_EXECUTABLE(cubemap2equirect src/cubemap2equirect.cpp ${SOURCE_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(cubemap2equirect ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

ADD_EXECUTABLE(equirect2persp src/equirect2persp.cpp ${SOURCE_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(equirect2persp ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

ADD_EXECUTABLE(cubemap2persp src/cubemap2persp.cpp ${SOURCE_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(cubemap2persp ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})