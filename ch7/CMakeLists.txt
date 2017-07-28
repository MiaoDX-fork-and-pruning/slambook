cmake_minimum_required( VERSION 2.8 )
project( vo1 )

message(STATUS "In vo1 project:")

#set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_BUILD_TYPE "Debug" ) 

if (WIN32)
message(STATUS "Going to define _WINDOWS:")
add_definitions(-D_WINDOWS) # `#ifdef _WINDOWS` in timeutil.h
set( CMAKE_CXX_FLAGS "-std=c++11" )
set( EIGEN_INCLUDE_DIR "C:/tools/eigen/" )
else()
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set( EIGEN_INCLUDE_DIR "/usr/include/eigen3/" )
endif()



# 添加cmake模块以使用g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

# find_package( OpenCV 3.1 REQUIRED )
# find_package( OpenCV REQUIRED ) # use this if in OpenCV2 
include( $ENV{OpenCV3_DIR}/OpenCVConfig.cmake ) # https://github.com/district10/cmake-templates/issues/4
find_package( G2O REQUIRED )
#find_package( CSparse REQUIRED ) # Not necessary since there is `EXTERNAL` in g2o

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
message(STATUS "Debug mode:")
set(G2O_LIBS ${G2O_LIBS_DEBUG})
else()
message(STATUS "Release mode:")
set(G2O_LIBS ${G2O_LIBS_RELEASE})
endif()
message(STATUS "G2O_INCLUDE_DIR: ${G2O_INCLUDE_DIR}")
#message(STATUS "G2O_LIBS_RELEASE: ${G2O_LIBS_RELEASE}") 
#message(STATUS "G2O_LIBS_DEBUG: ${G2O_LIBS_DEBUG}") 
message(STATUS "G2O_LIBS: ${G2O_LIBS}") 

#message(STATUS "CSPARSE_INCLUDE_DIR: ${CSPARSE_INCLUDE_DIR}")  
#message(STATUS "CSPARSE_LIBRARY: ${CSPARSE_LIBRARY}")  

message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")  
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")  



include_directories( 
    ${OpenCV_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIR}
    ${G2O_INCLUDE_DIR}/EXTERNAL/csparse #${CSPARSE_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}
)



add_executable( feature_extraction feature_extraction.cpp  )
target_link_libraries( feature_extraction ${OpenCV_LIBS} )

# add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp extra.cpp ) # use this if in OpenCV2 
add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp )
target_link_libraries( pose_estimation_2d2d ${OpenCV_LIBS} )

# add_executable( triangulation triangulation.cpp extra.cpp) # use this if in opencv2 
add_executable( triangulation triangulation.cpp )
target_link_libraries( triangulation ${OpenCV_LIBS} )

add_executable( pose_estimation_3d2d pose_estimation_3d2d.cpp )
target_link_libraries( pose_estimation_3d2d 
   ${OpenCV_LIBS}
   #${CSPARSE_LIBRARY}
   ${G2O_LIBS}
)

add_executable( pose_estimation_3d3d pose_estimation_3d3d.cpp )
target_link_libraries( pose_estimation_3d3d 
   ${OpenCV_LIBS}   
   #${CSPARSE_LIBRARY}
   ${G2O_LIBS}
) 

