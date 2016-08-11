#include <AIToolbox/MDP/SparseExperience.hpp>

#include <boost/python.hpp>

void exportMDPSparseExperience() {
    using namespace AIToolbox::MDP;
    using namespace boost::python;

    class_<SparseExperience>{"SparseExperience",

         "This class keeps track of registered events and rewards.\n"
         "\n"
         "This class is a simple logger of events. It keeps track of both\n"
         "the number of times a particular transition has happened, and the\n"
         "total reward gained in any particular transition. However, it\n"
         "does not record each event separately (i.e. you can't extract\n"
         "the results of a particular transition in the past).\n"
         "\n"
         "The difference between this class and the MDP.Experience class is\n"
         "that this class stores recorded events in sparse matrices. This\n"
         "results in very high space savings when the state space of the\n"
         "environment being logged is very high but only a small subset of\n"
         "the states are really possible, at the cost of some efficiency\n"
         "(possibly offset by cache savings).", no_init}

        .def(init<size_t, size_t>(
                 "Basic constructor.\n"
                 "\n"
                 "@param s The number of states of the world.\n"
                 "@param a The number of actions available to the agent."
        , (arg("self"), "s", "a")))

        .def("record",          &SparseExperience::record,
                 "This function adds a new event to the recordings.\n"
                 "\n"
                 "@param s     Old state.\n"
                 "@param a     Performed action.\n"
                 "@param s1    New state.\n"
                 "@param rew   Obtained reward."
        , (arg("self"), "s", "a", "s1", "rew"))

        .def("reset",           &SparseExperience::reset,
                 "This function resets all experienced rewards and transitions."
        , (arg("self")))

        .def("getVisits",       &SparseExperience::getVisits,
                 "This function returns the current recorded visits for a transitions.\n"
                 "\n"
                 "@param s     Old state.\n"
                 "@param a     Performed action.\n"
                 "@param s1    New state."
        , (arg("self"), "s", "a", "s1"))

        .def("getVisitsSum",    &SparseExperience::getVisitsSum,
                 "This function returns the number of transitions recorded that start with the specified state and action.\n"
                 "\n"
                 "@param s     The initial state.\n"
                 "@param a     Performed action.\n"
                 "\n"
                 "@return The total number of transitions that start with the specified state-action pair."
        , (arg("self"), "s", "a"))

        .def("getReward",       &SparseExperience::getReward,
                 "This function returns the cumulative rewards obtained from a specific transition.\n"
                 "\n"
                 "@param s     Old state.\n"
                 "@param a     Performed action.\n"
                 "@param s1    New state."
        , (arg("self"), "s", "a", "s1"))

        .def("getRewardSum",    &SparseExperience::getRewardSum,
                 "This function returns the total reward obtained from transitions that start with the specified state and action.\n"
                 "\n"
                 "@param s     The initial state.\n"
                 "@param a     Performed action.\n"
                 "\n"
                 "@return The total number of transitions that start with the specified state-action pair."
        , (arg("self"), "s", "a"))

        .def("getS",            &SparseExperience::getS,
                 "This function returns the number of states of the world."
        , (arg("self")))

        .def("getA",            &SparseExperience::getA,
                 "This function returns the number of available actions to the agent."
        , (arg("self")));
}
