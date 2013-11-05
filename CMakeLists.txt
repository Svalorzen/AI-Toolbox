cmake_minimum_required (VERSION 2.6)

project (MDPToolbox)

ADD_DEFINITIONS(
    -std=c++11 # Or -std=c++0x
    # Other flags
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${MDPToolbox_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${MDPToolbox_BINARY_DIR})

include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory (${PROJECT_SOURCE_DIR}/src)
add_subdirectory (${PROJECT_SOURCE_DIR}/test)
