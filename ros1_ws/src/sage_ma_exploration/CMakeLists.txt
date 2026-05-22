cmake_minimum_required(VERSION 3.0.2)
project(sage_ma_exploration)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  nav_msgs
  roscpp
  rospy
  std_msgs
  visualization_msgs
)

add_message_files(
  FILES
  CellUpdate.msg
  EdgeDelta.msg
  GraphDelta.msg
  Layer2Delta.msg
  NodeDelta.msg
  StateDigest.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    message_runtime
    nav_msgs
    roscpp
    rospy
    std_msgs
    visualization_msgs
)

catkin_install_python(PROGRAMS
  scripts/operator_pipeline_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
