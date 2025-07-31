cmake_minimum_required(VERSION 3.10.0)
project(ML_GINS_Calibration VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find packages
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenMP REQUIRED)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

find_package(GTSAM QUIET)
if(NOT GTSAM_FOUND)
    message(WARNING "GTSAM not found. Some targets may not be available.")
endif()

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR})

# Find system manif library
find_package(manif REQUIRED)
message(STATUS "Using system manif library")

# Include directories
include_directories(include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add subdirectories
add_subdirectory(src)

# Copy configuration files to build directory
file(COPY config DESTINATION ${CMAKE_BINARY_DIR})
