#include <AIToolbox/MDP/Experience.hpp>

#include <boost/python.hpp>

void exportExperience() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    class_<Experience>{"Experience", init<size_t, size_t>()}
        .def("getS",            &Experience::getS)
        .def("getA",            &Experience::getA)
        .def("record",          &Experience::record)
        .def("reset",           &Experience::reset)
        .def("getVisits",       &Experience::getVisits)
        .def("getVisitsSum",    &Experience::getVisitsSum)
        .def("getReward",       &Experience::getReward)
        .def("getRewardSum",    &Experience::getRewardSum);
}
