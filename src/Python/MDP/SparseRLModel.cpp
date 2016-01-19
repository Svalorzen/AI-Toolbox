#include <AIToolbox/MDP/SparseExperience.hpp>
#include <AIToolbox/MDP/SparseRLModel.hpp>

#include <boost/python.hpp>

using SparseRLModelBinded = AIToolbox::MDP::SparseRLModel<AIToolbox::MDP::SparseExperience>;

void exportSparseRLModel() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    class_<SparseRLModelBinded>{"SparseRLModel", init<const SparseExperience &, optional<double, bool>>()}
        .def("setDiscount",                 &SparseRLModelBinded::setDiscount)
        .def("getS",                        &SparseRLModelBinded::getS)
        .def("getA",                        &SparseRLModelBinded::getA)
        .def("getDiscount",                 &SparseRLModelBinded::getDiscount)
        .def("getExperience",               &SparseRLModelBinded::getExperience, return_value_policy<reference_existing_object>())
        .def("sync",                        static_cast<void(SparseRLModelBinded::*)()>(&SparseRLModelBinded::sync))
        .def("sync",                        static_cast<void(SparseRLModelBinded::*)(size_t, size_t)>(&SparseRLModelBinded::sync))
        .def("sync",                        static_cast<void(SparseRLModelBinded::*)(size_t, size_t, size_t)>(&SparseRLModelBinded::sync))
        .def("sampleSR",                    &SparseRLModelBinded::sampleSR)
        .def("getTransitionProbability",    &SparseRLModelBinded::getTransitionProbability)
        .def("getExpectedReward",           &SparseRLModelBinded::getExpectedReward)
        .def("isTerminal",                  &SparseRLModelBinded::isTerminal);
}
