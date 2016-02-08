#include <AIToolbox/MDP/Algorithms/SARSA.hpp>

#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/RLModel.hpp>
#include <AIToolbox/MDP/SparseExperience.hpp>
#include <AIToolbox/MDP/SparseRLModel.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>

#include <boost/python.hpp>

void exportSARSA() {
    using namespace boost::python;
    using namespace AIToolbox::MDP;

    class_<SARSA>{"SARSA", init<size_t, size_t, optional<double, double>>()}
        .def(init<const RLModel<Experience>&, optional<double>>())
        .def(init<const SparseRLModel<SparseExperience>&, optional<double>>())
        .def(init<const Model&, optional<double>>())
        .def(init<const SparseModel&, optional<double>>())
        .def("setLearningRate",             &SARSA::setLearningRate)
        .def("getLearningRate",             &SARSA::getLearningRate)
        .def("setDiscount",                 &SARSA::setDiscount)
        .def("getDiscount",                 &SARSA::getDiscount)
        .def("getS",                        &SARSA::getS)
        .def("getA",                        &SARSA::getA)
        .def("stepUpdateQ",                 &SARSA::stepUpdateQ)
        .def("getQFunction",                &SARSA::getQFunction, return_internal_reference<>());
}
