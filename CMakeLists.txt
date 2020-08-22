cmake_minimum_required(VERSION 2.8)

project( main )

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package( OpenCV 4.0.0 REQUIRED )

include_directories(
    includes
    ${OpenCV_INCLUDE_DIRS}
)

file(GLOB LOCAL_SRC "src/*.cpp")
file(GLOB LOCAL_INC "includes/*.hpp")

add_executable(
    main
    src/main.cpp
    ${LOCAL_SRC}
)

target_link_libraries( 
    main
    ${OpenCV_LIBS} 
)