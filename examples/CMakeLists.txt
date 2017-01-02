if (MAKE_MDP)
    find_package(Eigen3 REQUIRED)
    include_directories(${EIGEN3_INCLUDE_DIR})

    add_executable(tiger_antelope MDP/tiger_antelope.cpp)
    target_link_libraries(tiger_antelope AIToolboxMDP)

    add_custom_command(
        OUTPUT  "${CMAKE_CURRENT_BINARY_DIR}/tiger_antelope.py"
        COMMAND cp
        ARGS    "${CMAKE_CURRENT_LIST_DIR}/MDP/tiger_antelope.py"
                "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Copying python example to example folder...")
    add_custom_target(tiger_antelope_py ALL
        DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/tiger_antelope.py")

    if (MAKE_PYTHON)
    endif()
endif()

if (MAKE_POMDP)
    add_executable(tiger_door POMDP/tiger_door.cpp)
    target_link_libraries(tiger_door AIToolboxMDP AIToolboxPOMDP)

    if (MAKE_PYTHON)
    endif()
endif()
