cmake_minimum_required(VERSION 3.1)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")
    message(STATUS "No build type selected, default to ${CMAKE_BUILD_TYPE}")
endif()

project(pagerankcpp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-Wall -O3")

add_subdirectory(pybind11)
pybind11_add_module(pagerankcpp pagerankcpp.cpp)

add_subdirectory(eigen)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
if(TARGET Eigen3::Eigen)
    target_link_libraries(pagerankcpp PUBLIC Eigen3::Eigen)
endif()
    
add_custom_command(TARGET pagerankcpp 
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:pagerankcpp> ${CMAKE_SOURCE_DIR})
