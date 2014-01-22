cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(cad_recognition)

find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

FIND_PACKAGE(VTK)
INCLUDE(${VTK_USE_FILE})

FIND_PACKAGE(Qt4 REQUIRED)


add_executable (cad_recognition cad_recognition.cpp)
target_link_libraries (cad_recognition ${PCL_LIBRARIES})