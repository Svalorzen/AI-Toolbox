cmake_minimum_required (VERSION 2.6)

project (AIToolbox)

ADD_DEFINITIONS(
    -std=c++11 # Or -std=c++0x
    # Other flags
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
# Output executable in main directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})


include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory (${PROJECT_SOURCE_DIR}/src)
add_subdirectory (${PROJECT_SOURCE_DIR}/test)
