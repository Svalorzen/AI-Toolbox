#include <AIToolbox/MDP/Algorithms/ValueIteration.hpp>

#include <AIToolbox/MDP/Types.hpp>
#include "../../Utils.hpp"

#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/RLModel.hpp>
#include <AIToolbox/MDP/SparseExperience.hpp>
#include <AIToolbox/MDP/SparseRLModel.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>

#include <boost/python.hpp>

template <typename M>
void exportValueIterationByModel(std::string className) {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    using V = ValueIteration<M>;

    class_<V>{("ValueIteration" + className).c_str(), init<unsigned, optional<double>>()}
        .def("setEpsilon",              &V::setEpsilon)
        .def("setHorizon",              &V::setHorizon)
        .def("getEpsilon",              &V::getEpsilon)
        .def("getEpsilon",              &V::getHorizon)
        .def("__call__",                &V::operator());
}

void exportValueIteration() {
    using namespace AIToolbox::MDP;

    exportValueIterationByModel<RLModel<Experience>>("RLModel");
    exportValueIterationByModel<SparseRLModel<SparseExperience>>("SparseRLModel");
    exportValueIterationByModel<Model>("Model");
    exportValueIterationByModel<SparseModel>("SparseModel");
    // Enable reading the return type
    TupleToPython<std::tuple<bool, ValueFunction, QFunction>>();
}
