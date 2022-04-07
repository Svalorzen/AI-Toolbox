cmake_minimum_required (VERSION 3.12) # CMP0069 NEW

project (AIToolbox LANGUAGES CXX)

# We define a series of variables for the user. They can be combined in order
# to build exactly what is needed:
#
# MAKE_ALL:      Builds all there is to build in the project, but Python.
# MAKE_LIB:      Builds the core C++ library
# MAKE_MDP:      Builds the core C++ MDP library
# MAKE_FMDP:     Builds the core C++ Factored MDP and MDP library
# MAKE_POMDP:    Builds the core C++ POMDP and MDP library
# MAKE_TESTS:    Builds the library's tests for the compiled core library
# MAKE_EXAMPLES: Builds the library's examples using the compiled core library
# MAKE_PYTHON:   Builds Python bindings for the compiled core library
# AI_PYTHON_VERSION: Selects Python version to use
# AI_LOGGING_ENABLED: Enables logging in the library.

# NOTE TO COMPILE ON WINDOWS:
#
# On Windows it is generally much less practical to actually look here for
# folders and things we need, so you WILL probably need to pass paths to the
# CMake call manually, and possibly even touch this script a bit.
#
# Some settings I found useful when compiling on Windows:
#
# -DCMAKE_GENERATOR_PLATFORM=x64
# -DBOOST_LIBRARYDIR
#
# You may also want to force Boost to compile statically; you can change that
# by uncommenting two lines in this script that you can find below (grep for
# Boost and static).

##############################
##  CMake helper functions  ##
##############################

function(append value)
    foreach(variable ${ARGN})
        set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
    endforeach(variable)
endfunction()

##############################
##     Project Settings     ##
##############################

# Set default cmake build type to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are:
          Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

# Give default value to all option settings (0 if they were not set)
# - The only one we don't preset here if unset is AI_PYTHON_VERSION, since we do that later.
foreach(v MAKE_ALL;MAKE_LIB;MAKE_MDP;MAKE_FMDP;MAKE_POMDP;MAKE_TESTS;MAKE_EXAMPLES;MAKE_PYTHON;AI_LOGGING_ENABLED)
    if (NOT DEFINED ${v} OR NOT ${${v}})
        set(${v} 0)
    endif()
endforeach(v)

set(DEFAULT_BUILD 0)

# Default is to build everything
if (NOT MAKE_ALL AND NOT MAKE_LIB AND NOT MAKE_MDP AND NOT MAKE_FMDP AND NOT MAKE_POMDP)
    set(DEFAULT_BUILD 1)
    set(MAKE_ALL    1)
endif()

if (MAKE_ALL)
    set(MAKE_LIB        1)
    set(MAKE_TESTS      1)
    set(MAKE_EXAMPLES   1)
endif()

if(MAKE_LIB)
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

##############################
## Compiler/Linker Settings ##
##############################

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT WIN32)
    add_definitions(
        -Wall
        -Wextra
    )
endif()

# Check for Link Time Optimizations with this compiler
include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_ERROR)

##############################
##       Dependencies       ##
##############################

set(BOOST_VERSION_REQUIRED 1.67)
set(EIGEN_VERSION_REQUIRED 3.2.92)

# Optional to force Boost to use static libraries. Can be useful on Windows.
#
# set(Boost_USE_STATIC_LIBS ON)
# add_definitions(-DBOOST_PYTHON_STATIC_LIB)

find_package(Boost ${BOOST_VERSION_REQUIRED} REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

find_package(Eigen3 ${EIGEN_VERSION_REQUIRED} REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

find_package(LpSolve REQUIRED)
include_directories(SYSTEM ${LPSOLVE_INCLUDE_DIR})

if (MAKE_PYTHON)
    # If we build Python's shared library, then all libraries we link into it
    # have to be compiled with -fPIC
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    set(MAKE_PYTHON     1)
    set(Python_USE_STATIC_LIBS 0)

    # Set a default to AI_PYTHON_VERSION to void breaking the next if. In any
    # case, if Python is not explicitly set to 2, we'll pick what is there
    # anyway (so this doesn't actually force Python 3).
    if (NOT AI_PYTHON_VERSION)
        set(AI_PYTHON_VERSION 3)
    endif()

    if (${AI_PYTHON_VERSION} EQUAL 2)
        find_package(Python2 COMPONENTS Interpreter Development REQUIRED)
        # Rename libs so we can keep the rest of the script independent of version
        set(Python_LIBRARIES ${Python2_LIBRARIES})
        set(Python_VERSION_MAJOR ${Python2_VERSION_MAJOR})
        set(Python_VERSION_MINOR ${Python2_VERSION_MINOR})
        set(Python_INCLUDE_DIRS ${Python2_INCLUDE_DIRS})
        set(Python_EXECUTABLE ${Python2_EXECUTABLE})
    else()
        # This tries to find out which version of Python we should be targeting depending
        # on which interpreter is found, but prefers Python 3 so if we selected that we should be alright.
        find_package(Python COMPONENTS Interpreter Development REQUIRED)
        set(AI_PYTHON_VERSION ${Python_VERSION_MAJOR})
    endif()

    find_package(Boost ${BOOST_VERSION_REQUIRED} COMPONENTS python${Python_VERSION_MAJOR}${Python_VERSION_MINOR} REQUIRED)
    # Set independent name for lib
    set(BOOST_PYTHON_LIBRARY_NAME "Boost_PYTHON${Python_VERSION_MAJOR}${Python_VERSION_MINOR}_LIBRARIES")

    include_directories(SYSTEM ${Python_INCLUDE_DIRS})
endif()

if (MAKE_TESTS)
    find_package(Boost ${BOOST_VERSION_REQUIRED} COMPONENTS unit_test_framework REQUIRED)
endif()

##############################
##      User Feedback       ##
##############################

if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColorReset "${Esc}[m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(Magenta     "${Esc}[35m")
  set(Cyan        "${Esc}[36m")
  set(White       "${Esc}[37m")
endif()

# These strings contain the user feedback on what things we are actually building.
set(MAP_MAKE_ALL           "Building everything      (-DMAKE_ALL=${MAKE_ALL})")
set(MAP_MAKE_LIB           "Building library         (-DMAKE_LIB=${MAKE_LIB})")
set(MAP_MAKE_MDP           "Building MDP             (-DMAKE_MDP=${MAKE_MDP})")
set(MAP_MAKE_POMDP         "Building POMDP           (-DMAKE_POMDP=${MAKE_POMDP})")
set(MAP_MAKE_FMDP          "Building Factored MDP    (-DMAKE_FMDP=${MAKE_FMDP})")
set(MAP_MAKE_TESTS         "Building Tests           (-DMAKE_TESTS=${MAKE_TESTS})")
set(MAP_MAKE_EXAMPLES      "Building Examples        (-DMAKE_EXAMPLES=${MAKE_EXAMPLES})")
set(MAP_MAKE_PYTHON        "Building Python bindings (-DMAKE_PYTHON=${MAKE_PYTHON})")
if (${MAKE_PYTHON})
    set(MAP_MAKE_PYTHON "${MAP_MAKE_PYTHON}\n  - Selected Python ${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}    (-DAI_PYTHON_VERSION=${AI_PYTHON_VERSION})")
endif()
set(MAP_AI_LOGGING_ENABLED "Enabled runtime logging  (-DAI_LOGGING_ENABLED=${AI_LOGGING_ENABLED})")

# Actual feedback print.
message("")

message("### AI_TOOLBOX SETTINGS ###")
if (DEFAULT_BUILD)
    message("- No project build options provided, defaulting to MAKE_ALL..")
endif()

message(STATUS "CMAKE_BUILD_TYPE:   " ${CMAKE_BUILD_TYPE})
if( LTO_SUPPORTED )
    message(STATUS "IPO / LTO enabled")
else()
    message(STATUS "IPO / LTO not supported: <${LTO_ERROR}>")
endif()

foreach(v MAKE_ALL;MAKE_LIB;MAKE_MDP;MAKE_FMDP;MAKE_POMDP;MAKE_TESTS;MAKE_EXAMPLES;MAKE_PYTHON;AI_LOGGING_ENABLED)
    set(N "${Green}✓${ColorReset} ")
    if (NOT ${${v}})
        set(N "${Cyan}✗${ColorReset} NOT ")
    endif()

    message("${N}${MAP_${v}}")
endforeach(v)

message("")

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
