#include <AIToolbox/Bandit/Experience.hpp>

#include <boost/python.hpp>

void exportBanditExperience() {
    using namespace AIToolbox::Bandit;
    using namespace boost::python;

    class_<Experience>{"Experience",

         "This class computes averages and counts for a Bandit problem.\n"
         "\n"
         "This class can be used to compute the averages and counts for all\n"
         "actions in a Bandit problem over time.", no_init}

        .def(init<size_t>(
                 "Basic constructor.\n"
                 "\n"
                 "@param A The size of the action space."
        , (arg("self"), "A")))

        .def("record",     &Experience::record,
                 "This function updates the the reward matrix and counts.\n"
                 "\n"
                 "@param a The action taken.\n"
                 "@param rew The reward obtained."
        , (arg("self"), "a", "rew"))

        .def("reset",           &Experience::reset,
                 "This function resets the QFunction and counts to zero."
        , (arg("self")))

        .def("getTimesteps",    &Experience::getTimesteps,
                 "This function returns the number of times that record has been called."
        , (arg("self")))

        .def("getA",            &Experience::getA,
                 "This function returns the size of the action space."
        , (arg("self")))

        .def("getRewardMatrix", &Experience::getRewardMatrix, return_internal_reference<>(),
                 "This function returns a reference to the internal QFunction."
        , (arg("self")))

        .def("getM2Matrix",     &Experience::getM2Matrix, return_internal_reference<>(),
                 "This function returns a reference to the internal QFunction."
        , (arg("self")))

        .def("getVisitsTable",  &Experience::getVisitsTable, return_internal_reference<>(),
                 "This function returns a reference for the counts for the actions"
        , (arg("self")));
}
