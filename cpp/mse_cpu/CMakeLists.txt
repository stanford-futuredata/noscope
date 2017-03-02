project(mse_cpu)

cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
  list(APPEND ThirdParty_LIBS ${OpenCV_LIBS})
  include_directories(${OpenCV_INCLUDE_DIRS})
endif(OpenCV_FOUND)

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -march=native -std=c++1y -O3 -ftree-vectorize")

set(${PROJECT_NAME}_SRC mse_cpu.cpp)
add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_SRC} ${${PROJECT_NAME}_HDR})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
