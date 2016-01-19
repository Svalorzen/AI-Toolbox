#include <AIToolbox/MDP/SparseModel.hpp>

#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/RLModel.hpp>
#include <AIToolbox/MDP/SparseExperience.hpp>
#include <AIToolbox/MDP/SparseRLModel.hpp>
#include <AIToolbox/MDP/Model.hpp>

#include <boost/python.hpp>

void exportSparseModel() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    class_<SparseModel>{"SparseModel", init<size_t, size_t, optional<double>>()}
        .def(init<const Model &>())
        .def(init<const SparseModel &>())
        .def(init<const RLModel<Experience> &>())
        .def(init<const SparseRLModel<SparseExperience> &>())
        .def("setDiscount",                 &Model::setDiscount)
        .def("setTransitionFunction",       &Model::setTransitionFunction<std::vector<std::vector<std::vector<double>>>>)
        .def("setRewardFunction",           &Model::setRewardFunction<std::vector<std::vector<std::vector<double>>>>)
        .def("getS",                        &Model::getS)
        .def("getA",                        &Model::getA)
        .def("getDiscount",                 &Model::getDiscount)
        .def("sampleSR",                    &Model::sampleSR)
        .def("getTransitionProbability",    &Model::getTransitionProbability)
        .def("getExpectedReward",           &Model::getExpectedReward)
        .def("isTerminal",                  &Model::isTerminal);
}
