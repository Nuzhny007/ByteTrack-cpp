cmake_minimum_required(VERSION 3.14)
project(bytetrack)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -MMD -Wall -Wextra -Winit-self")

set(Boost_USE_STATIC_LIBS    ON)
set(Boost_USE_MULTITHREADED  ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost)
include_directories(${Boost_INCLUDE_DIRS})

set(EIGEN_VERSION 3.3)
find_package(Eigen3 ${EIGEN_VERSION} QUIET)
if(NOT EIGEN3_FOUND)
  set(BUILD_TESTING OFF CACHE INTERNAL "")
  FetchContent_Declare(eigen
    GIT_REPOSITORY  https://gitlab.com/libeigen/eigen.git
    GIT_TAG         ${EIGEN_VERSION}
    GIT_SHALLOW     ON)
  FetchContent_MakeAvailable(eigen)
  unset(BUILD_TESTING CACHE)
endif()

add_library(${PROJECT_NAME} STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/BYTETracker.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KalmanFilter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/lapjv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Object.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Rect.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/STrack.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

# Build the tests if the 'BUILD_BYTETRACK_TEST' variable is set to 'ON'
set(BUILD_BYTETRACK_TEST OFF CACHE BOOL "The flag whether to build the tests or not")
if(BUILD_BYTETRACK_TEST)
  enable_testing()

  find_package(OpenCV REQUIRED)
  include_directories(${OpenCV_INCLUDE_DIRS})

  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    message(FATAL_ERROR "GTest not found.")
  endif()

  add_executable(${PROJECT_NAME}_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_BYTETracker.cpp
  )

  target_link_libraries(${PROJECT_NAME}_test
    ${PROJECT_NAME}
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
    )

  gtest_discover_tests(${PROJECT_NAME}_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/YOLOX_ncnn_palace)
endif()
