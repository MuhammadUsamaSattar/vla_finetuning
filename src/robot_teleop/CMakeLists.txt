cmake_minimum_required(VERSION 3.20)
project(robot_teleop)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(MUJOCO_DIR "$ENV{HOME}/mujoco")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(Boost REQUIRED COMPONENTS filesystem system)
#find_package()
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

add_executable(data_collection src/data_collection.cpp)
add_library(controller src/controller.cpp)
add_library(mujoco_sim src/mujoco_sim.cpp)
add_library(hdf5_saver src/hdf5_saver.cpp)

target_include_directories(mujoco_sim PUBLIC
  ${MUJOCO_DIR}/include
  include/
)
target_include_directories(data_collection PRIVATE
  ${MUJOCO_DIR}/include
  include/
)
target_include_directories(controller PRIVATE
  ${MUJOCO_DIR}/include
  include/
)
target_include_directories(hdf5_saver PUBLIC 
  ${MUJOCO_DIR}/include 
  include
)

target_link_libraries(mujoco_sim PUBLIC
  hdf5_saver
  ${MUJOCO_DIR}/lib/libmujoco.so
  rclcpp::rclcpp
  ament_index_cpp::ament_index_cpp
  glfw
  Eigen3::Eigen
  controller
)
target_link_libraries(data_collection
  mujoco_sim
  controller
)
target_link_libraries(hdf5_saver PUBLIC
  Boost::filesystem
  Boost::system
  HDF5::HDF5
  ${MUJOCO_DIR}/lib/libmujoco.so
)

install(TARGETS 
  hdf5_saver
  mujoco_sim
  data_collection
  controller
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY mujoco/
  DESTINATION share/${PROJECT_NAME}/mujoco
)

install(DIRECTORY include/
  DESTINATION include/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
