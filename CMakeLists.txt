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

##############################
##  CMake helper functions  ##
##############################

include(CheckCXXCompilerFlag)

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

add_definitions(
    -std=c++17
    -Wall
    -Wextra
)

# Do release linker and option optimizations
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    # Use ld.gold if it is available and isn't disabled explicitly
    option(USE_LD_GOLD "Use GNU gold linker" ON)
    if (USE_LD_GOLD)
        check_cxx_compiler_flag("-fuse-ld=gold" GOLD_IS_AVAILABLE)
        if (NOT GOLD_IS_AVAILABLE)
            message(WARNING "GNU gold linker isn't available, using the default system linker.")
        endif ()
    endif ()

    # Per-compiler type optional options
    if     ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        # using Clang

        if (GOLD_IS_AVAILABLE)
            APPEND("-fuse-ld=gold" CMAKE_EXE_LINKER_FLAGS
            CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)

            # Cannot enable lto in Clang since it explodes - it needs a gold version
            # which supports plugins, and we'd have to check for that. AND it should
            # have the clang-gold plugin available for use anyway.
            #
            # add_definitions( -flto )
        endif()

    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"  )
        # using GCC

        # Add link time optimizations
        add_definitions( -flto )
        append("-flto" CMAKE_EXE_LINKER_FLAGS CMAKE
            CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
        # Use gcc-ar so that we can optimize the static libraries
        set(CMAKE_AR "gcc-ar")
        set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> qcs <TARGET> <LINK_FLAGS> <OBJECTS>")
        set(CMAKE_CXX_ARCHIVE_FINISH true)
        # Try to use gold as it is faster if possible
        if (GOLD_IS_AVAILABLE)
            append("-fuse-ld=gold" CMAKE_EXE_LINKER_FLAGS
            CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
        endif()

    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        # using Intel C++
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" )
        # using Visual Studio C++
    endif()
endif()

##############################
##     Project Settings     ##
##############################

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
message("Build type: " ${CMAKE_BUILD_TYPE})
foreach(v MAKE_MDP;MAKE_POMDP;MAKE_PYTHON;MAKE_TESTS;MAKE_EXAMPLES)
    if (${${v}})
        message(${MAP_${v}})
    endif()
endforeach(v)
message("")
