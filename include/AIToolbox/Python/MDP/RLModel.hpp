#ifndef AI_TOOLBOX_PYTHON_MDP_RLMODEL_HEADER_FILE
#define AI_TOOLBOX_PYTHON_MDP_RLMODEL_HEADER_FILE

#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/RLModel.hpp>

#include <boost/python.hpp>

using RLModelBinded = AIToolbox::MDP::RLModel<AIToolbox::MDP::Experience>;

void exportRLModel() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    class_<RLModelBinded>{"RLModel", init<const Experience &, optional<double, bool>>()}
        .def("setDiscount",                 &RLModelBinded::setDiscount)
        .def("getS",                        &RLModelBinded::getS)
        .def("getA",                        &RLModelBinded::getA)
        .def("getDiscount",                 &RLModelBinded::getDiscount)
        .def("getExperience",               &RLModelBinded::getExperience, return_value_policy<reference_existing_object>())
        .def("sync",                        static_cast<void(RLModelBinded::*)()>(&RLModelBinded::sync))
        .def("sync",                        static_cast<void(RLModelBinded::*)(size_t, size_t)>(&RLModelBinded::sync))
        .def("sync",                        static_cast<void(RLModelBinded::*)(size_t, size_t, size_t)>(&RLModelBinded::sync))
        .def("sampleSR",                    &RLModelBinded::sampleSR)
        .def("getTransitionProbability",    &RLModelBinded::getTransitionProbability)
        .def("getExpectedReward",           &RLModelBinded::getExpectedReward)
        .def("isTerminal",                  &RLModelBinded::isTerminal);
}

#endif
