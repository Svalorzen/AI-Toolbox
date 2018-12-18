#include <AIToolbox/Bandit/Algorithms/RollingAverage.hpp>

#include <boost/python.hpp>

void exportBanditRollingAverage() {
    using namespace AIToolbox::Bandit;
    using namespace boost::python;

    class_<RollingAverage>{"RollingAverage",

         "This class computes averages and counts for a Bandit problem.\n"
         "\n"
         "This class can be used to compute the averages and counts for all\n"
         "actions in a Bandit problem over time.", no_init}

        .def(init<size_t>(
                 "Basic constructor.\n"
                 "\n"
                 "@param A The size of the action space."
        , (arg("self"), "A")))

        .def("stepUpdateQ",     &RollingAverage::stepUpdateQ,
                 "This function updates the QFunction and counts.\n"
                 "\n"
                 "@param a The action taken.\n"
                 "@param rew The reward obtained."
        , (arg("self"), "a", "rew"))

        .def("reset",           &RollingAverage::reset,
                 "This function resets the QFunction and counts to zero."
        , (arg("self")))

        .def("getA",            &RollingAverage::getA,
                 "This function returns the size of the action space."
        , (arg("self")))

        .def("getQFunction",    &RollingAverage::getQFunction, return_internal_reference<>(),
                 "This function returns a reference to the internal QFunction."
        , (arg("self")))

        .def("getCounts",       &RollingAverage::getCounts, return_internal_reference<>(),
                 "This function returns a reference for the counts for the actions"
        , (arg("self")));
}
