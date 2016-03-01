#include <AIToolbox/MDP/Algorithms/MCTS.hpp>

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
void exportMCTSByModel(std::string className) {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    using V = MCTS<M>;

    size_t (V::*sampleAction1)(size_t, unsigned) = &V::sampleAction;
    size_t (V::*sampleAction2)(size_t, size_t, unsigned) = &V::sampleAction;

    class_<V>{("PrioritizedSweeping" + className).c_str(), init<const M&, unsigned, double>()}
        .def("sampleAction",            sampleAction1)
        .def("sampleAction",            sampleAction2)
        .def("setIterations",           &V::setIterations)
        .def("getIterations",           &V::getIterations)
        .def("setExploration",          &V::setExploration)
        .def("getExploration",          &V::getExploration)
        .def("getModel",                &V::getModel,   return_value_policy<reference_existing_object>());
}

void exportMCTS() {
    using namespace AIToolbox::MDP;

    exportMCTSByModel<RLModel<Experience>>("RLModel");
    exportMCTSByModel<SparseRLModel<SparseExperience>>("SparseRLModel");
    exportMCTSByModel<Model>("Model");
    exportMCTSByModel<SparseModel>("SparseModel");
}
