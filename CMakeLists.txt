cmake_minimum_required (VERSION 3.0)

project (AIToolbox)

# We define a series of variables for the user. They can be combined in order
# to build exactly what is needed:
#
# MAKE_ALL:      Builds all there is to build in the project
# MAKE_LIB:      Builds the core C++ library
# MAKE_MDP:      Builds the core C++ MDP library
# MAKE_POMDP:    Builds the core C++ POMDP and MDP library
# MAKE_PYTHON:   Builds Python bindings for the compiled core library
# MAKE_TESTS:    Builds the library's tests for the compiled core library
# MAKE_EXAMPLES: Builds the library's examples using the compiled core library

add_definitions(
    -std=c++11 # Or -std=c++0x
    -Wall
    -Wextra
    # Other flags
)

# Set default cmake build type to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are:
          Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

# Default is to build everything
IF (NOT MAKE_ALL AND NOT MAKE_LIB AND NOT MAKE_MDP AND NOT MAKE_POMDP)
    set(MAKE_ALL    1)
endif()

if (MAKE_ALL)
    set(MAKE_MDP        1)
    set(MAKE_POMDP      1)
    set(MAKE_PYTHON     1)
    set(MAKE_TESTS      1)
    set(MAKE_EXAMPLES   1)
elseif (MAKE_LIB)
    set(MAKE_MDP        1)
    set(MAKE_POMDP      1)
elseif (MAKE_POMDP)
    set(MAKE_MDP        1)
endif()

# For additional Find library scripts
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# Add library directories
include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory (${PROJECT_SOURCE_DIR}/src)

# If enabled, add tests
if (MAKE_TESTS)
    include(CTest)
    add_subdirectory (${PROJECT_SOURCE_DIR}/test)
endif()

# If enabled, add Python bindings
if (MAKE_EXAMPLES)
    add_subdirectory (${PROJECT_SOURCE_DIR}/examples)
endif()

# Print what we're actually doing
set(MAP_MAKE_MDP        "# Building MDP")
set(MAP_MAKE_POMDP      "# Building POMDP")
set(MAP_MAKE_PYTHON     "# Building Python bindings")
set(MAP_MAKE_TESTS      "# Building Tests")
set(MAP_MAKE_EXAMPLES   "# Building Examples")

message("")
foreach(v MAKE_MDP;MAKE_POMDP;MAKE_PYTHON;MAKE_TESTS;MAKE_EXAMPLES)
    if (${${v}})
        message(${MAP_${v}})
    endif()
endforeach(v)
message("")
