cmake_minimum_required(VERSION 3.8)
project(slam_optimization_tutorial)

# ==============================================================================
# Compiler Options
# ==============================================================================
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ==============================================================================
# C++ Standard
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==============================================================================
# 3rd Party Dependencies
# ==============================================================================
find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(Ceres REQUIRED)

# ==============================================================================
# Executable
# ==============================================================================
add_executable(transformation src/transformation.cpp)

target_include_directories(transformation PRIVATE
    ${GTSAM_INCLUDE_DIR}
    ${CERES_INCLUDE_DIRS}
)

target_link_libraries(transformation
  Eigen3::Eigen
  gtsam
  Ceres::ceres
)