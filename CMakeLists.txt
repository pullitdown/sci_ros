cmake_minimum_required(VERSION 3.0.2)
project(sci_ros)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
  sensor_msgs
  image_transport
)
 
catkin_package(
#INCLUDE_DIRS include
#LIBRARIES lively_slam
CATKIN_DEPENDS 
roscpp rospy std_msgs cv_bridge sensor_msgs image_transport 
#DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
)

