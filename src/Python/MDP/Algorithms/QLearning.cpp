#include <AIToolbox/MDP/Algorithms/QLearning.hpp>

#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/RLModel.hpp>
#include <AIToolbox/MDP/SparseExperience.hpp>
#include <AIToolbox/MDP/SparseRLModel.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>

#include <boost/python.hpp>

template <typename M>
void exportQLearningByModel(std::string className) {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    using Q = QLearning<M>;

    class_<Q>{("QLearning" + className).c_str(), init<const M&, optional<double>>()}
        .def("setLearningRate",             &Q::setLearningRate)
        .def("getLearningRate",             &Q::getLearningRate)
        .def("stepUpdateQ",                 &Q::stepUpdateQ)
        .def("getQFunction",                &Q::getQFunction, return_internal_reference<>())
        .def("getModel",                    &Q::getModel, return_value_policy<reference_existing_object>());
;
}


void exportQLearning() {
    using namespace AIToolbox::MDP;

    exportQLearningByModel<RLModel<Experience>>("RLModel");
    exportQLearningByModel<SparseRLModel<SparseExperience>>("SparseRLModel");
    exportQLearningByModel<Model>("Model");
    exportQLearningByModel<SparseModel>("SparseModel");
}
