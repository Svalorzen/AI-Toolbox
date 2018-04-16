#include "../Utils.hpp"

#include <AIToolbox/MDP/Types.hpp>

void exportMDPTypes() {
    using namespace AIToolbox;
    using namespace boost::python;

    // MDP Value Function
    class_<MDP::ValueFunction>("ValueFunction")
        .def_readwrite("values", &MDP::ValueFunction::values)
        .def_readwrite("actions", &MDP::ValueFunction::actions);

    // Results of sampleSR
    TupleToPython<std::tuple<size_t, double>>();

    // We export method-specific types here since it is possible other methods
    // could use them in the future, and we don't want to duplicate them if
    // possible.

    // ValueIteration return value
    TupleToPython<std::tuple<double, MDP::ValueFunction, MDP::QFunction>>();
}
