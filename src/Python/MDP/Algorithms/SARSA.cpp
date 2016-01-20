#include <AIToolbox/MDP/Algorithms/SARSA.hpp>

#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/RLModel.hpp>
#include <AIToolbox/MDP/SparseExperience.hpp>
#include <AIToolbox/MDP/SparseRLModel.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>

#include <boost/python.hpp>

template <typename M>
void exportSARSAByModel(std::string className) {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    using S = SARSA<M>;

    class_<S>{("SARSA" + className).c_str(), init<const M&, optional<double>>()}
        .def("setLearningRate",             &S::setLearningRate)
        .def("getLearningRate",             &S::getLearningRate)
        .def("stepUpdateQ",                 &S::stepUpdateQ)
        .def("getQFunction",                &S::getQFunction, return_internal_reference<>())
        .def("getModel",                    &S::getModel, return_value_policy<reference_existing_object>());
}


void exportSARSA() {
    using namespace AIToolbox::MDP;

    exportSARSAByModel<RLModel<Experience>>("RLModel");
    exportSARSAByModel<SparseRLModel<SparseExperience>>("SparseRLModel");
    exportSARSAByModel<Model>("Model");
    exportSARSAByModel<SparseModel>("SparseModel");
}
