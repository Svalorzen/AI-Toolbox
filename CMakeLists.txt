cmake_minimum_required (VERSION 2.6)

project (AIToolbox)

ADD_DEFINITIONS(
    -std=c++11 # Or -std=c++0x
    -Wall
    -Wextra
    # Other flags
)

# Default is to build everything
if (NOT MAKE_ALL AND NOT MAKE_MDP AND NOT MAKE_POMDP)
    set(MAKE_ALL    1)
endif()

if (MAKE_ALL)
    set(MAKE_MDP    1)
    set(MAKE_POMDP  1)
elseif(MAKE_POMDP)
    set(MAKE_MDP    1)
endif()

# For additional Find library scripts
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
# All executables are tests
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/test)

enable_testing()

include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory (${PROJECT_SOURCE_DIR}/src)
add_subdirectory (${PROJECT_SOURCE_DIR}/test)
