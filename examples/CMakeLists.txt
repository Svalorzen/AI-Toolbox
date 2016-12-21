if (MAKE_MDP)
    find_package(Eigen3 REQUIRED)
    include_directories(${EIGEN3_INCLUDE_DIR})

    add_executable(tiger_antelope MDP/tiger_antelope.cpp)
    target_link_libraries(tiger_antelope AIToolboxMDP)

    if (MAKE_PYTHON)
    endif()
endif()

if (MAKE_POMDP)
    if (MAKE_PYTHON)
    endif()
endif()
