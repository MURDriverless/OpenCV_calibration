cmake_minimum_required(VERSION 2.8)

project( main )

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