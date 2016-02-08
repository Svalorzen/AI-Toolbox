#include <AIToolbox/MDP/Algorithms/PrioritizedSweeping.hpp>

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
void exportPrioritizedSweepingByModel(std::string className) {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    using V = PrioritizedSweeping<M>;

    class_<V>{("PrioritizedSweeping" + className).c_str(), init<const M&, optional<double, unsigned>>()}
        .def("stepUpdateQ",             &V::stepUpdateQ)
        .def("batchUpdateQ",            &V::batchUpdateQ)
        .def("setQueueThreshold",       &V::setQueueThreshold)
        .def("getQueueThreshold",       &V::getQueueThreshold)
        .def("setN",                    &V::setN)
        .def("getN",                    &V::getN)
        .def("getQueueLength",          &V::getQueueLength)
        .def("getModel",                &V::getModel,   return_value_policy<reference_existing_object>())
        .def("getQFunction",            &V::getQFunction, return_internal_reference<>())
        .def("setQFunction",            &V::setQFunction)
        .def("getValueFunction",        &V::getValueFunction, return_internal_reference<>());
}

void exportPrioritizedSweeping() {
    using namespace AIToolbox::MDP;

    exportPrioritizedSweepingByModel<RLModel<Experience>>("RLModel");
    exportPrioritizedSweepingByModel<SparseRLModel<SparseExperience>>("SparseRLModel");
    exportPrioritizedSweepingByModel<Model>("Model");
    exportPrioritizedSweepingByModel<SparseModel>("SparseModel");
}
