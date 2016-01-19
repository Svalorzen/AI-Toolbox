#include <AIToolbox/MDP/SparseExperience.hpp>

#include <boost/python.hpp>

void exportSparseExperience() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    class_<SparseExperience>{"SparseExperience", init<size_t, size_t>()}
        .def("getS",            &SparseExperience::getS)
        .def("getA",            &SparseExperience::getA)
        .def("record",          &SparseExperience::record)
        .def("reset",           &SparseExperience::reset)
        .def("getVisits",       &SparseExperience::getVisits)
        .def("getVisitsSum",    &SparseExperience::getVisitsSum)
        .def("getReward",       &SparseExperience::getReward)
        .def("getRewardSum",    &SparseExperience::getRewardSum);
}
