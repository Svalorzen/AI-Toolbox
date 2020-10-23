#include <AIToolbox/MDP/Environments/Utils/GridWorld.hpp>

#include <boost/python.hpp>

void exportMDPGridWorld() {
    using namespace boost::python;
    using namespace AIToolbox::MDP;

    class_<GridWorld>{"GridWorld",

         "This class represents a simple rectangular gridworld.", no_init }

        .def(init<size_t, size_t, optional<bool>>(
                 "Basic constructor.\n"
                 "\n"
                 "@param width The number of columns in the world.\n"
                 "@param height The number of rows in the world.\n"
                 "@param torus Whether to join the edges of the grid as in a torus."
        , (arg("self"), "width", "height", "torus")))

        .def("getWidth",            &GridWorld::getWidth,
                 "This function returns the width of the GridWorld."
        , (arg("self")))

        .def("getHeight",           &GridWorld::getHeight,
                 "This function returns the height of the GridWorld."
        , (arg("self")))

        .def("isTorus",             &GridWorld::isTorus,
                 "This function returns whether the GridWorld represents a torus."
        , (arg("self")))

        .def("getS",                &GridWorld::getS,
                 "This function returns the number of cells in the grid."
        , (arg("self")));
}
