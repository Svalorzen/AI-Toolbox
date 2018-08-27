cmake_minimum_required(VERSION 3.9) # CMP0069 NEW

if (MAKE_MDP)
    add_executable(tiger_antelope MDP/tiger_antelope.cpp)
    target_link_libraries(tiger_antelope AIToolboxMDP)
    set_target_properties(tiger_antelope PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})

    if (MAKE_PYTHON)
        add_custom_command(
            OUTPUT  "${CMAKE_CURRENT_BINARY_DIR}/tiger_antelope.py"
            COMMAND cp
            ARGS    "${CMAKE_CURRENT_LIST_DIR}/MDP/tiger_antelope.py"
                    "${CMAKE_CURRENT_BINARY_DIR}"
            COMMENT "Copying python tiger_antelope to example folder..."
            DEPENDS "${CMAKE_CURRENT_LIST_DIR}/MDP/tiger_antelope.py")
        add_custom_target(tiger_antelope_py ALL
            DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/tiger_antelope.py")
    endif()
endif()

if (MAKE_POMDP)
    add_executable(tiger_door POMDP/tiger_door.cpp)
    target_link_libraries(tiger_door AIToolboxMDP AIToolboxPOMDP)
    set_target_properties(tiger_door PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})

    if (MAKE_PYTHON)
        add_custom_command(
            OUTPUT  "${CMAKE_CURRENT_BINARY_DIR}/tiger_door.py"
            COMMAND cp
            ARGS    "${CMAKE_CURRENT_LIST_DIR}/POMDP/tiger_door.py"
                    "${CMAKE_CURRENT_BINARY_DIR}"
            COMMENT "Copying python tiger_door to example folder..."
            DEPENDS "${CMAKE_CURRENT_LIST_DIR}/POMDP/tiger_door.py")
        add_custom_target(tiger_door_py ALL
            DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/tiger_door.py")
    endif()
endif()
