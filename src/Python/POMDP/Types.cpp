#include "../Utils.hpp"

#include <AIToolbox/POMDP/Types.hpp>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

void exportPOMDPTypes() {
    using namespace AIToolbox;
    using namespace boost::python;

    // Results of POMDP policy with horizon
    TupleToPython<std::tuple<size_t, size_t>>();
    // Results of sampleSOR
    TupleToPython<std::tuple<size_t, size_t, double>>();

    // POMDP Value Function
    TupleToPython<POMDP::VEntry>();
    // FIXME: Here we set the NoProxy parameter of the vector_indexing_suite to
    // true, otherwise there is a weird bug where the VEntry cannot be seen in
    // Python if it is extracted from the VList. This unfortunately means that
    // the VList has some unintuitive behaviour if one tries to edit it
    // inplace, but we cannot really do anything about that at the moment.
    class_<POMDP::VList>{"VList"}
        .def(vector_indexing_suite<POMDP::VList, true>());
    class_<POMDP::ValueFunction>{"POMDP_VFun"}
        .def(vector_indexing_suite<POMDP::ValueFunction>());

    // We export method-specific types here since it is possible other methods
    // could use them in the future, and we don't want to duplicate them if
    // possible.

    // IncrementalPruning return value
    TupleToPython<std::tuple<double, POMDP::ValueFunction>>();
}
