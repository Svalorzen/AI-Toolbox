cmake_minimum_required (VERSION 3.9) # CMP0069 NEW

project (AIToolbox LANGUAGES CXX)

# We define a series of variables for the user. They can be combined in order
# to build exactly what is needed:
#
# MAKE_ALL:      Builds all there is to build in the project
# MAKE_LIB:      Builds the core C++ library
# MAKE_MDP:      Builds the core C++ MDP library
# MAKE_FMDP:     Builds the core C++ Factored MDP and MDP library
# MAKE_POMDP:    Builds the core C++ POMDP and MDP library
# MAKE_PYTHON:   Builds Python bindings for the compiled core library
# MAKE_TESTS:    Builds the library's tests for the compiled core library
# MAKE_EXAMPLES: Builds the library's examples using the compiled core library

##############################
##  CMake helper functions  ##
##############################

function(append value)
    foreach(variable ${ARGN})
        set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
    endforeach(variable)
endfunction()

##############################
## Compiler/Linker Settings ##
##############################

# Set default cmake build type to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are:
          Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_definitions(
    -Wall
    -Wextra
)

# Check for Link Time Optimizations with this compiler
include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_ERROR)

if( LTO_SUPPORTED )
    message(STATUS "IPO / LTO enabled")
else()
    message(STATUS "IPO / LTO not supported: <${LTO_ERROR}>")
endif()

##############################
##     Project Settings     ##
##############################

# Default is to build everything
if (NOT MAKE_ALL AND NOT MAKE_LIB AND NOT MAKE_MDP AND NOT MAKE_FMDP AND NOT MAKE_POMDP)
    set(MAKE_ALL    1)
endif()

if (MAKE_ALL)
    set(MAKE_MDP        1)
    set(MAKE_FMDP       1)
    set(MAKE_POMDP      1)
    set(MAKE_PYTHON     1)
    set(MAKE_TESTS      1)
    set(MAKE_EXAMPLES   1)
elseif (MAKE_LIB)
    set(MAKE_MDP        1)
    set(MAKE_FMDP       1)
    set(MAKE_POMDP      1)
elseif (MAKE_FMDP)
    set(MAKE_MDP        1)
elseif (MAKE_POMDP)
    set(MAKE_MDP        1)
endif()

# Check whether to enable logging
if (${AI_LOGGING_ENABLED})
    add_definitions(-DAI_LOGGING_ENABLED)
    set(LOGGING_STATUS "ENABLED")
else()
    set(LOGGING_STATUS "DISABLED")
endif()

# For additional Find library scripts
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# Print what we're actually doing
set(MAP_MAKE_MDP        "# Building MDP")
set(MAP_MAKE_FMDP       "# Building Factored MDP")
set(MAP_MAKE_POMDP      "# Building POMDP")
set(MAP_MAKE_PYTHON     "# Building Python bindings")
set(MAP_MAKE_TESTS      "# Building Tests")
set(MAP_MAKE_EXAMPLES   "# Building Examples")

message("")
message("Build type: " ${CMAKE_BUILD_TYPE})
message("Logging is " ${LOGGING_STATUS})
foreach(v MAKE_MDP;MAKE_FMDP;MAKE_POMDP;MAKE_PYTHON;MAKE_TESTS;MAKE_EXAMPLES)
    if (${${v}})
        message(${MAP_${v}})
    endif()
endforeach(v)
message("")

##############################
##       Dependencies       ##
##############################

set(BOOST_VERSION_REQUIRED 1.54)
set(EIGEN_VERSION_REQUIRED 3.2.92)

find_package(Boost ${BOOST_VERSION_REQUIRED} REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

find_package(Eigen3 ${EIGEN_VERSION_REQUIRED} REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

find_package(LpSolve REQUIRED)
include_directories(SYSTEM ${LPSOLVE_INCLUDE_DIR})

if (MAKE_PYTHON)
    # Try to find out which version of Python we should be targeting depending
    # on which interpreter is found. If the version has been selected
    # explicitly we search for it directly.
    find_package(PythonInterp ${PYTHON_VERSION})
    set(PYTHON_VERSION ${PYTHON_VERSION_MAJOR})

    if (${PYTHON_VERSION} EQUAL 3)
        find_package(PythonLibs 3 REQUIRED)
        find_package(Boost ${BOOST_VERSION_REQUIRED} COMPONENTS python3 REQUIRED)
        set(BOOST_PYTHON_LIBRARY_NAME "Boost_PYTHON3_LIBRARY")
    else()
        find_package(PythonLibs 2.7 EXACT REQUIRED)
        find_package(Boost ${BOOST_VERSION_REQUIRED} COMPONENTS python REQUIRED)
        set(BOOST_PYTHON_LIBRARY_NAME "Boost_PYTHON_LIBRARY")
    endif()

    include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS})
endif()

if (MAKE_TESTS)
    find_package(Boost ${BOOST_VERSION_REQUIRED} COMPONENTS unit_test_framework REQUIRED)
endif()

##############################
##      Project Start       ##
##############################

# Add library directories
include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/src)

# If enabled, add tests
if (MAKE_TESTS)
    include(CTest)
    add_subdirectory(${PROJECT_SOURCE_DIR}/test)
endif()

# If enabled, add Python bindings
if (MAKE_EXAMPLES)
    add_subdirectory(${PROJECT_SOURCE_DIR}/examples)
endif()
