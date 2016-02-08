#include <AIToolbox/MDP/Algorithms/QLearning.hpp>

#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/RLModel.hpp>
#include <AIToolbox/MDP/SparseExperience.hpp>
#include <AIToolbox/MDP/SparseRLModel.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>

#include <boost/python.hpp>

void exportQLearning() {
    using namespace boost::python;
    using namespace AIToolbox::MDP;

    class_<QLearning>{"QLearning", init<size_t, size_t, optional<double, double>>()}
        .def(init<const RLModel<Experience>&, optional<double>>())
        .def(init<const SparseRLModel<SparseExperience>&, optional<double>>())
        .def(init<const Model&, optional<double>>())
        .def(init<const SparseModel&, optional<double>>())
        .def("setLearningRate",             &QLearning::setLearningRate)
        .def("getLearningRate",             &QLearning::getLearningRate)
        .def("setDiscount",                 &QLearning::setDiscount)
        .def("getDiscount",                 &QLearning::getDiscount)
        .def("getS",                        &QLearning::getS)
        .def("getA",                        &QLearning::getA)
        .def("stepUpdateQ",                 &QLearning::stepUpdateQ)
        .def("getQFunction",                &QLearning::getQFunction, return_internal_reference<>());
}
